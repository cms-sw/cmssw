/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/**
 * Modified to cache pinned host allocations by Matti Kortelainen
 */

/******************************************************************************
 * Simple caching allocator for pinned host memory allocations. The allocator is
 * thread-safe.
 ******************************************************************************/

#pragma once

#include <set>
#include <map>
#include <cmath>

#include <cub/util_namespace.cuh>
#include <cub/util_debug.cuh>
#include <cub/host/mutex.cuh>

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/**
 * \addtogroup UtilMgmt
 * @{
 */


/******************************************************************************
 * CachingHostAllocator (host use)
 ******************************************************************************/

/**
 * \brief A simple caching allocator pinned host memory allocations.
 *
 * \par Overview
 * The allocator is thread-safe.  It behaves as follows:
 *
 * I presume the CUDA stream-safeness is not useful as to read/write
 * from/to the pinned host memory one needs to synchronize anyway. The
 * difference wrt. device memory is that in the CPU all operations to
 * the device memory are scheduled via the CUDA stream, while for the
 * host memory one can perform operations directly.
 *
 * \par
 * - Allocations are categorized and cached by bin size.  A new allocation request of
 *   a given size will only consider cached allocations within the corresponding bin.
 * - Bin limits progress geometrically in accordance with the growth factor
 *   \p bin_growth provided during construction.  Unused host allocations within
 *   a larger bin cache are not reused for allocation requests that categorize to
 *   smaller bin sizes.
 * - Allocation requests below (\p bin_growth ^ \p min_bin) are rounded up to
 *   (\p bin_growth ^ \p min_bin).
 * - Allocations above (\p bin_growth ^ \p max_bin) are not rounded up to the nearest
 *   bin and are simply freed when they are deallocated instead of being returned
 *   to a bin-cache.
 * - %If the total storage of cached allocations  will exceed
 *   \p max_cached_bytes, allocations are simply freed when they are
 *   deallocated instead of being returned to their bin-cache.
 *
 * \par
 * For example, the default-constructed CachingHostAllocator is configured with:
 * - \p bin_growth          = 8
 * - \p min_bin             = 3
 * - \p max_bin             = 7
 * - \p max_cached_bytes    = 6MB - 1B
 *
 * \par
 * which delineates five bin-sizes: 512B, 4KB, 32KB, 256KB, and 2MB
 * and sets a maximum of 6,291,455 cached bytes
 *
 */
struct CachingHostAllocator
{

    //---------------------------------------------------------------------
    // Constants
    //---------------------------------------------------------------------

    /// Out-of-bounds bin
    static const unsigned int INVALID_BIN = (unsigned int) -1;

    /// Invalid size
    static const size_t INVALID_SIZE = (size_t) -1;

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

    /// Invalid device ordinal
    static const int INVALID_DEVICE_ORDINAL = -1;

    //---------------------------------------------------------------------
    // Type definitions and helper types
    //---------------------------------------------------------------------

    /**
     * Descriptor for pinned host memory allocations
     */
    struct BlockDescriptor
    {
        void*           d_ptr;              // Host pointer
        size_t          bytes;              // Size of allocation in bytes
        unsigned int    bin;                // Bin enumeration
        int             device;             // device ordinal
        cudaStream_t    associated_stream;  // Associated associated_stream
        cudaEvent_t     ready_event;        // Signal when associated stream has run to the point at which this block was freed

        // Constructor (suitable for searching maps for a specific block, given its pointer)
        BlockDescriptor(void *d_ptr) :
            d_ptr(d_ptr),
            bytes(0),
            bin(INVALID_BIN),
            device(INVALID_DEVICE_ORDINAL),
            associated_stream(0),
            ready_event(0)
        {}

        // Constructor (suitable for searching maps for a range of suitable blocks)
        BlockDescriptor() :
            d_ptr(NULL),
            bytes(0),
            bin(INVALID_BIN),
            device(INVALID_DEVICE_ORDINAL),
            associated_stream(0),
            ready_event(0)
        {}

        // Comparison functor for comparing host pointers
        static bool PtrCompare(const BlockDescriptor &a, const BlockDescriptor &b)
        {
            return (a.d_ptr < b.d_ptr);
        }

        // Comparison functor for comparing allocation sizes
        static bool SizeCompare(const BlockDescriptor &a, const BlockDescriptor &b)
        {
            return (a.bytes < b.bytes);
        }
    };

    /// BlockDescriptor comparator function interface
    typedef bool (*Compare)(const BlockDescriptor &, const BlockDescriptor &);

    class TotalBytes {
    public:
        size_t free;
        size_t live;
        TotalBytes() { free = live = 0; }
    };

    /// Set type for cached blocks (ordered by size)
    typedef std::multiset<BlockDescriptor, Compare> CachedBlocks;

    /// Set type for live blocks (ordered by ptr)
    typedef std::multiset<BlockDescriptor, Compare> BusyBlocks;


    //---------------------------------------------------------------------
    // Utility functions
    //---------------------------------------------------------------------

    /**
     * Integer pow function for unsigned base and exponent
     */
    static unsigned int IntPow(
        unsigned int base,
        unsigned int exp)
    {
        unsigned int retval = 1;
        while (exp > 0)
        {
            if (exp & 1) {
                retval = retval * base;        // multiply the result by the current base
            }
            base = base * base;                // square the base
            exp = exp >> 1;                    // divide the exponent in half
        }
        return retval;
    }


    /**
     * Round up to the nearest power-of
     */
    void NearestPowerOf(
        unsigned int    &power,
        size_t          &rounded_bytes,
        unsigned int    base,
        size_t          value)
    {
        power = 0;
        rounded_bytes = 1;

        if (value * base < value)
        {
            // Overflow
            power = sizeof(size_t) * 8;
            rounded_bytes = size_t(0) - 1;
            return;
        }

        while (rounded_bytes < value)
        {
            rounded_bytes *= base;
            power++;
        }
    }


    //---------------------------------------------------------------------
    // Fields
    //---------------------------------------------------------------------

    cub::Mutex      mutex;              /// Mutex for thread-safety

    unsigned int    bin_growth;         /// Geometric growth factor for bin-sizes
    unsigned int    min_bin;            /// Minimum bin enumeration
    unsigned int    max_bin;            /// Maximum bin enumeration

    size_t          min_bin_bytes;      /// Minimum bin size
    size_t          max_bin_bytes;      /// Maximum bin size
    size_t          max_cached_bytes;   /// Maximum aggregate cached bytes

    const bool      skip_cleanup;       /// Whether or not to skip a call to FreeAllCached() when destructor is called.  (The CUDA runtime may have already shut down for statically declared allocators)
    bool            debug;              /// Whether or not to print (de)allocation events to stdout

    TotalBytes      cached_bytes;       /// Aggregate cached bytes
    CachedBlocks    cached_blocks;      /// Set of cached pinned host allocations available for reuse
    BusyBlocks      live_blocks;        /// Set of live pinned host allocations currently in use

#endif // DOXYGEN_SHOULD_SKIP_THIS

    //---------------------------------------------------------------------
    // Methods
    //---------------------------------------------------------------------

    /**
     * \brief Constructor.
     */
    CachingHostAllocator(
        unsigned int    bin_growth,                             ///< Geometric growth factor for bin-sizes
        unsigned int    min_bin             = 1,                ///< Minimum bin (default is bin_growth ^ 1)
        unsigned int    max_bin             = INVALID_BIN,      ///< Maximum bin (default is no max bin)
        size_t          max_cached_bytes    = INVALID_SIZE,     ///< Maximum aggregate cached bytes (default is no limit)
        bool            skip_cleanup        = false,            ///< Whether or not to skip a call to \p FreeAllCached() when the destructor is called (default is to deallocate)
        bool            debug               = false)            ///< Whether or not to print (de)allocation events to stdout (default is no stderr output)
    :
        bin_growth(bin_growth),
        min_bin(min_bin),
        max_bin(max_bin),
        min_bin_bytes(IntPow(bin_growth, min_bin)),
        max_bin_bytes(IntPow(bin_growth, max_bin)),
        max_cached_bytes(max_cached_bytes),
        skip_cleanup(skip_cleanup),
        debug(debug),
        cached_blocks(BlockDescriptor::SizeCompare),
        live_blocks(BlockDescriptor::PtrCompare)
    {}


    /**
     * \brief Default constructor.
     *
     * Configured with:
     * \par
     * - \p bin_growth          = 8
     * - \p min_bin             = 3
     * - \p max_bin             = 7
     * - \p max_cached_bytes    = (\p bin_growth ^ \p max_bin) * 3) - 1 = 6,291,455 bytes
     *
     * which delineates five bin-sizes: 512B, 4KB, 32KB, 256KB, and 2MB and
     * sets a maximum of 6,291,455 cached bytes
     */
    CachingHostAllocator(
        bool skip_cleanup = false,
        bool debug = false)
    :
        bin_growth(8),
        min_bin(3),
        max_bin(7),
        min_bin_bytes(IntPow(bin_growth, min_bin)),
        max_bin_bytes(IntPow(bin_growth, max_bin)),
        max_cached_bytes((max_bin_bytes * 3) - 1),
        skip_cleanup(skip_cleanup),
        debug(debug),
        cached_blocks(BlockDescriptor::SizeCompare),
        live_blocks(BlockDescriptor::PtrCompare)
    {}


    /**
     * \brief Sets the limit on the number bytes this allocator is allowed to cache
     *
     * Changing the ceiling of cached bytes does not cause any allocations (in-use or
     * cached-in-reserve) to be freed.  See \p FreeAllCached().
     */
    void SetMaxCachedBytes(
        size_t max_cached_bytes)
    {
        // Lock
        mutex.Lock();

        if (debug) _CubLog("Changing max_cached_bytes (%lld -> %lld)\n", (long long) this->max_cached_bytes, (long long) max_cached_bytes);

        this->max_cached_bytes = max_cached_bytes;

        // Unlock
        mutex.Unlock();
    }


    /**
     * \brief Provides a suitable allocation of pinned host memory for the given size.
     *
     * Once freed, the allocation becomes available immediately for reuse.
     */
    cudaError_t HostAllocate(
        void            **d_ptr,            ///< [out] Reference to pointer to the allocation
        size_t          bytes,              ///< [in] Minimum number of bytes for the allocation
        cudaStream_t    active_stream = 0)  ///< [in] The stream to be associated with this allocation
    {
        *d_ptr                          = NULL;
        int device                      = INVALID_DEVICE_ORDINAL;
        cudaError_t error               = cudaSuccess;

        if (CubDebug(error = cudaGetDevice(&device))) return error;

        // Create a block descriptor for the requested allocation
        bool found = false;
        BlockDescriptor search_key;
        search_key.device = device;
        search_key.associated_stream = active_stream;
        NearestPowerOf(search_key.bin, search_key.bytes, bin_growth, bytes);

        if (search_key.bin > max_bin)
        {
            // Bin is greater than our maximum bin: allocate the request
            // exactly and give out-of-bounds bin.  It will not be cached
            // for reuse when returned.
            search_key.bin      = INVALID_BIN;
            search_key.bytes    = bytes;
        }
        else
        {
            // Search for a suitable cached allocation: lock
            mutex.Lock();

            if (search_key.bin < min_bin)
            {
                // Bin is less than minimum bin: round up
                search_key.bin      = min_bin;
                search_key.bytes    = min_bin_bytes;
            }

            // Iterate through the range of cached blocks in the same bin
            CachedBlocks::iterator block_itr = cached_blocks.lower_bound(search_key);
            while ((block_itr != cached_blocks.end())
                    && (block_itr->bin == search_key.bin))
            {
                // To prevent races with reusing blocks returned by the host but still
                // in use for transfers, only consider cached blocks that are from an idle stream
                if(cudaEventQuery(block_itr->ready_event) != cudaErrorNotReady) {
                    // Reuse existing cache block.  Insert into live blocks.
                    found = true;
                    search_key = *block_itr;
                    search_key.associated_stream = active_stream;
                    live_blocks.insert(search_key);

                    // Remove from free blocks
                    cached_bytes.free -= search_key.bytes;
                    cached_bytes.live += search_key.bytes;

                    if (debug) _CubLog("\tHost reused cached block at %p (%lld bytes) for stream %lld (previously associated with stream %lld).\n",
                        search_key.d_ptr, (long long) search_key.bytes, (long long) search_key.associated_stream, (long long)  block_itr->associated_stream);

                    cached_blocks.erase(block_itr);

                    break;
                }
                block_itr++;
            }

            // Done searching: unlock
            mutex.Unlock();
        }

        // Allocate the block if necessary
        if (!found)
        {
            // Attempt to allocate
            // TODO: eventually support allocation flags
            if (CubDebug(error = cudaHostAlloc(&search_key.d_ptr, search_key.bytes, cudaHostAllocDefault)) == cudaErrorMemoryAllocation)
            {
                // The allocation attempt failed: free all cached blocks on device and retry
                if (debug) _CubLog("\tHost failed to allocate %lld bytes for stream %lld, retrying after freeing cached allocations",
                      (long long) search_key.bytes, (long long) search_key.associated_stream);

                error = cudaSuccess;    // Reset the error we will return
                cudaGetLastError();     // Reset CUDART's error

                // Lock
                mutex.Lock();

                // Iterate the range of free blocks
                CachedBlocks::iterator block_itr = cached_blocks.begin();

                while ((block_itr != cached_blocks.end()))
                {
                    // No need to worry about synchronization with the device: cudaFree is
                    // blocking and will synchronize across all kernels executing
                    // on the current device
  
                    // Free pinned host memory.
                    if (CubDebug(error = cudaFreeHost(block_itr->d_ptr))) break;
                    if (CubDebug(error = cudaEventDestroy(block_itr->ready_event))) break;

                    // Reduce balance and erase entry
                    cached_bytes.free -= block_itr->bytes;

                    if (debug) _CubLog("\tHost freed %lld bytes.\n\t\t  %lld available blocks cached (%lld bytes), %lld live blocks (%lld bytes) outstanding.\n",
                        (long long) block_itr->bytes, (long long) cached_blocks.size(), (long long) cached_bytes.free, (long long) live_blocks.size(), (long long) cached_bytes.live);

                    cached_blocks.erase(block_itr);

                    block_itr++;
                }

                // Unlock
                mutex.Unlock();

                // Return under error
                if (error) return error;

                // Try to allocate again
                if (CubDebug(error = cudaHostAlloc(&search_key.d_ptr, search_key.bytes, cudaHostAllocDefault))) return error;
            }

            // Create ready event
            if (CubDebug(error = cudaEventCreateWithFlags(&search_key.ready_event, cudaEventDisableTiming)))
                return error;

            // Insert into live blocks
            mutex.Lock();
            live_blocks.insert(search_key);
            cached_bytes.live += search_key.bytes;
            mutex.Unlock();

            if (debug) _CubLog("\tHost allocated new host block at %p (%lld bytes associated with stream %lld).\n",
                      search_key.d_ptr, (long long) search_key.bytes, (long long) search_key.associated_stream);
        }

        // Copy host pointer to output parameter
        *d_ptr = search_key.d_ptr;

        if (debug) _CubLog("\t\t%lld available blocks cached (%lld bytes), %lld live blocks outstanding(%lld bytes).\n",
            (long long) cached_blocks.size(), (long long) cached_bytes.free, (long long) live_blocks.size(), (long long) cached_bytes.live);

        return error;
    }


    /**
     * \brief Frees a live allocation of pinned host memory, returning it to the allocator.
     *
     * Once freed, the allocation becomes available immediately for reuse.
     */
    cudaError_t HostFree(
        void*           d_ptr)
    {
        int entrypoint_device           = INVALID_DEVICE_ORDINAL;
        cudaError_t error               = cudaSuccess;

        // Lock
        mutex.Lock();

        // Find corresponding block descriptor
        bool recached = false;
        BlockDescriptor search_key(d_ptr);
        BusyBlocks::iterator block_itr = live_blocks.find(search_key);
        if (block_itr != live_blocks.end())
        {
            // Remove from live blocks
            search_key = *block_itr;
            live_blocks.erase(block_itr);
            cached_bytes.live -= search_key.bytes;

            // Keep the returned allocation if bin is valid and we won't exceed the max cached threshold
            if ((search_key.bin != INVALID_BIN) && (cached_bytes.free + search_key.bytes <= max_cached_bytes))
            {
                // Insert returned allocation into free blocks
                recached = true;
                cached_blocks.insert(search_key);
                cached_bytes.free += search_key.bytes;

                if (debug) _CubLog("\tHost returned %lld bytes from associated stream %lld.\n\t\t %lld available blocks cached (%lld bytes), %lld live blocks outstanding. (%lld bytes)\n",
                    (long long) search_key.bytes, (long long) search_key.associated_stream, (long long) cached_blocks.size(),
                    (long long) cached_bytes.free, (long long) live_blocks.size(), (long long) cached_bytes.live);
            }
        }

        // Unlock
        mutex.Unlock();

        if (CubDebug(error = cudaGetDevice(&entrypoint_device))) return error;
        if (entrypoint_device != search_key.device) {
            if (CubDebug(error = cudaSetDevice(search_key.device))) return error;
        }

        if (recached) {
            // Insert the ready event in the associated stream (must have current device set properly)
            if (CubDebug(error = cudaEventRecord(search_key.ready_event, search_key.associated_stream))) return error;
        }
        else
        {
            // Free the allocation from the runtime and cleanup the event.
            if (CubDebug(error = cudaFreeHost(d_ptr))) return error;
            if (CubDebug(error = cudaEventDestroy(search_key.ready_event))) return error;

            if (debug) _CubLog("\tHost freed %lld bytes from associated stream %lld.\n\t\t  %lld available blocks cached (%lld bytes), %lld live blocks (%lld bytes) outstanding.\n",
                (long long) search_key.bytes, (long long) search_key.associated_stream, (long long) cached_blocks.size(), (long long) cached_bytes.free, (long long) live_blocks.size(), (long long) cached_bytes.live);
        }

        // Reset device
        if ((entrypoint_device != INVALID_DEVICE_ORDINAL) && (entrypoint_device != search_key.device))
        {
            if (CubDebug(error = cudaSetDevice(entrypoint_device))) return error;
        }

        return error;
    }


    /**
     * \brief Frees all cached pinned host allocations
     */
    cudaError_t FreeAllCached()
    {
        cudaError_t error         = cudaSuccess;
        int entrypoint_device     = INVALID_DEVICE_ORDINAL;
        int current_device        = INVALID_DEVICE_ORDINAL;

        mutex.Lock();

        while (!cached_blocks.empty())
        {
            // Get first block
            CachedBlocks::iterator begin = cached_blocks.begin();

            // Get entry-point device ordinal if necessary
            if (entrypoint_device == INVALID_DEVICE_ORDINAL)
            {
                if (CubDebug(error = cudaGetDevice(&entrypoint_device))) break;
            }

            // Set current device ordinal if necessary
            if (begin->device != current_device)
            {
                if (CubDebug(error = cudaSetDevice(begin->device))) break;
                current_device = begin->device;
            }

            // Free host memory
            if (CubDebug(error = cudaFreeHost(begin->d_ptr))) break;
            if (CubDebug(error = cudaEventDestroy(begin->ready_event))) break;

            // Reduce balance and erase entry
            cached_bytes.free -= begin->bytes;

            if (debug) _CubLog("\tHost freed %lld bytes.\n\t\t  %lld available blocks cached (%lld bytes), %lld live blocks (%lld bytes) outstanding.\n",
                (long long) begin->bytes, (long long) cached_blocks.size(), (long long) cached_bytes.free, (long long) live_blocks.size(), (long long) cached_bytes.live);

            cached_blocks.erase(begin);
        }

        mutex.Unlock();

        // Attempt to revert back to entry-point device if necessary
        if (entrypoint_device != INVALID_DEVICE_ORDINAL)
        {
            if (CubDebug(error = cudaSetDevice(entrypoint_device))) return error;
        }

        return error;
    }


    /**
     * \brief Destructor
     */
    ~CachingHostAllocator()
    {
        if (!skip_cleanup)
            FreeAllCached();
    }

};




/** @} */       // end group UtilMgmt

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

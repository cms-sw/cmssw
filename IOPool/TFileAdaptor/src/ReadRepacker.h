
/**
 * Repack a set of read requests from the ROOT layer to be optimized for the
 * storage layer.
 * 
 * The basic technique employed is to coalesce nearby, but not adjacent reads
 * into one larger read in the request to the storage system.  We will be
 * purposely over-reading from storage.
 *
 * The read-coalescing is done because the vector reads are typically
 * unrolled server-side in a "dumb" fashion, with OS read-ahead disabled.
 * The coalescing actually decreases the number of requests sent to disk;
 * important, as ROOT I/O is typically latency bound.
 *
 * The complexity here is in the fact that we must have buffer space to hold
 * the extra bytes from the storage system, even through they're going to be
 * discarded.
 * 
 * The approach is to reuse the ROOT buffer as temporary holding space, plus
 * a small, fixed-size "spare buffer".  So, in the worst-case, we will use
 * about 256KB of extra buffer space.  The read-coalesce algorithm is greedy,
 * so we can't provide an a-priori estimate on how many extra I/O transactions
 * will be sent to the storage (compared to vector-reads with no coalescing).
 * Tests currently indicate that this approach usually causes zero to one
 * additional I/O transaction to occur.
 */

#include <vector>

# include "Utilities/StorageFactory/interface/IOPosBuffer.h"

class ReadRepacker {

public:

// Returns the number of input buffers it was able to pack into the IO operation.
int
pack(long long int    *pos,   // An array of file offsets to read.
     int              *len,   // An array of lengths to read.
     int               nbuf,  // Size of the pos and len array.
     char             *buf,   // Temporary buffer to hold I/O result.
     IOSize            buffer_size); // Size of the temporary buffer.

void
unpack(char *buf);  // Buffer to unpack the I/O results into.  Not the temporayr buffer and result buffer may overlap.

std::vector<IOPosBuffer> & iov() { return m_iov; } // Returns the IO vector, optimized for storage.

IOSize bufferUsed() const {return m_buffer_used;} // Returns the total amount of space in the temp buffer used.
IOSize extraBytes() const {return m_extra_bytes;} // Returns the number of extra bytes to be issued to the I/O system
                                                   // Note that (buffer_used - extra_bytes) should equal the number of "real" bytes serviced.
IOSize realBytesProcessed() const {return m_buffer_used-m_extra_bytes;} // Return the number of bytes of the input request that would be processed by the IO vector

// Two reads distanced by less than READ_COALESCE_SIZE will turn into one
// large read.
static const IOSize TEMPORARY_BUFFER_SIZE = 256 * 1024;

// A read larger than BIG_READ_SIZE will not be coalesced.
static const IOSize READ_COALESCE_SIZE = 32 * 1024;

// The size of the temporary holding buffer for read-coalescing.
static const IOSize BIG_READ_SIZE = 256 * 1024;

private:

int packInternal(long long int *pos, int *len, int nbuf, char *buf, IOSize buffer_size); // Heart of the implementation of Pack; because we pack up to 2 buffers,
                                                                                    // its easier to break the method into two.

void reset(unsigned int nbuf); // Reset all the internal counters and arrays.  Resize arrays to be about nbuf long.

std::vector<int>         m_idx_to_iopb;        // Mapping from idx in the input array to the iopb in the IO vector
std::vector<int>         m_idx_to_iopb_offset; // Mapping from idx in the input array to the data offset in the results of the iopb.
std::vector<IOPosBuffer> m_iov;                // Vector of IO for the storage system to perform.
int                     *m_len;                // Pointed to the array of read sizes.
IOSize                   m_buffer_used;        // Bytes in the temporary buffer used.
IOSize                   m_extra_bytes;        // Number of bytes read from storage that will be discarded.
std::vector<char>        m_spare_buffer;       // The spare buffer; allocated if we cannot fit the I/O results into the ROOT buffer.

};


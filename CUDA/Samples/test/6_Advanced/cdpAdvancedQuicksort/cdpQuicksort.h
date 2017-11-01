// Definitions for GPU quicksort

#ifndef QUICKSORT_H
#define QUICKSORT_H

#define QSORT_BLOCKSIZE_SHIFT   9
#define QSORT_BLOCKSIZE         (1 << QSORT_BLOCKSIZE_SHIFT)
#define BITONICSORT_LEN         1024            // Must be power of 2!
#define QSORT_MAXDEPTH          16              // Will force final bitonic stage at depth QSORT_MAXDEPTH+1

////////////////////////////////////////////////////////////////////////////////
// The algorithm uses several variables updated by using atomic operations.
////////////////////////////////////////////////////////////////////////////////
typedef struct __align__(128) qsortAtomicData_t
{
    volatile unsigned int lt_offset;    // Current output offset for <pivot
    volatile unsigned int gt_offset;    // Current output offset for >pivot
    volatile unsigned int sorted_count; // Total count sorted, for deciding when to launch next wave
    volatile unsigned int index;        // Ringbuf tracking index. Can be ignored if not using ringbuf.
} qsortAtomicData;

////////////////////////////////////////////////////////////////////////////////
// A ring-buffer for rapid stack allocation
////////////////////////////////////////////////////////////////////////////////
typedef struct qsortRingbuf_t
{
    volatile unsigned int head;         // Head pointer - we allocate from here
    volatile unsigned int tail;         // Tail pointer - indicates last still-in-use element
    volatile unsigned int count;        // Total count allocated
    volatile unsigned int max;          // Max index allocated
    unsigned int stacksize;             // Wrap-around size of buffer (must be power of 2)
    volatile void *stackbase;           // Pointer to the stack we're allocating from
} qsortRingbuf;

// Stack elem count must be power of 2!
#define QSORT_STACK_ELEMS   1*1024*1024 // One million stack elements is a HUGE number.

__global__ void qsort_warp(unsigned *indata, unsigned *outdata, unsigned int len, qsortAtomicData *atomicData, qsortRingbuf *ringbuf, unsigned int source_is_indata, unsigned int depth);
__global__ void bitonicsort(unsigned *indata, unsigned *outdata, unsigned int offset, unsigned int len);
__global__ void big_bitonicsort(unsigned *indata, unsigned *outdata, unsigned *backbuf, unsigned int offset, unsigned int len);

#endif // QUICKSORT_H

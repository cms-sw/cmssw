
#include <cassert>
#include <string.h>

#include "ReadRepacker.h"

/**
   Given a list of offsets and positions, pack them into a vector of IOPosBuffer (an "IO Vector").
   This function will coalesce reads that are within READ_COALESCE_SIZE into a IOPosBuffer.
   This function will not create an IO vector whose summed buffer size is larger than TEMPORARY_BUFFER_SIZE. 
   The IOPosBuffer in iov all point to a location inside buf.
    
   @param pos: An array of file offsets, nbuf long.
   @param len: An array of offset length, nbuf long.
   @param nbuf: Number of buffers to pack.
   @param buf: Location of temporary buffer for the results of the storage request.
   @param buffer_size: Size of the temporary buffer.

   Returns the number of entries of the original array packed into iov.
 */
int
ReadRepacker::pack(long long int *pos, int *len, int nbuf, char *buf, IOSize buffer_size)
{
  reset(nbuf);
  m_len = len; // Record the len array so we can later unpack.

  // Determine the buffer to use for the initial packing.
  char * tmp_buf;
  IOSize tmp_size;
  if (buffer_size < TEMPORARY_BUFFER_SIZE) {
        m_spare_buffer.resize(TEMPORARY_BUFFER_SIZE);
        tmp_buf = &m_spare_buffer[0];
        tmp_size = TEMPORARY_BUFFER_SIZE;
  } else {
        tmp_buf = buf;
        tmp_size = buffer_size;
  } 
  
  int pack_count = packInternal(pos, len, nbuf, tmp_buf, tmp_size);

  if ((nbuf - pack_count > 0) &&  // If there is remaining work..
      (tmp_buf != &m_spare_buffer[0]) &&    // and the spare buffer isn't already used
      ((IOSize)len[pack_count] < TEMPORARY_BUFFER_SIZE)) { // And the spare buffer is big enough to hold at least one read.

    // Verify the spare is allocated.
    // If tmp_buf != &m_spare_buffer[0] before, it certainly won't after.
    m_spare_buffer.resize(TEMPORARY_BUFFER_SIZE);

    // If there are remaining chunks and we aren't already using the spare
    // buffer, try using that too.
    // This clutters up the code badly, but could save a network round-trip.
    pack_count += packInternal(&pos[pack_count], &len[pack_count], nbuf-pack_count,
                               &m_spare_buffer[0], TEMPORARY_BUFFER_SIZE);

  }

  return pack_count;
}

int
ReadRepacker::packInternal(long long int *pos, int *len, int nbuf, char *buf, IOSize buffer_size)
{
  if (nbuf == 0) {
    return 0;
  }

  // Handle case 1 separately to make the for-loop cleaner.
  int iopb_offset = m_iov.size();
  // Because we re-use the buffer from ROOT, we are guarantee this iopb will
  // fit.
  assert(static_cast<IOSize>(len[0]) <= buffer_size);
  IOPosBuffer iopb(pos[0], buf, len[0]);
  m_idx_to_iopb.push_back(iopb_offset);
  m_idx_to_iopb_offset.push_back(0);

  IOSize buffer_used = len[0];
  int idx;
  for (idx=1; idx < nbuf; idx++) {
    if (buffer_used + len[idx] > buffer_size) {
      // No way we can include this chunk in the read buffer
      break;
    }

    IOOffset extra_bytes_signed = (idx == 0) ? 0 : ((pos[idx] - iopb.offset()) - iopb.size()); assert(extra_bytes_signed >= 0);
    IOSize   extra_bytes = static_cast<IOSize>(extra_bytes_signed);

    if (((static_cast<IOSize>(len[idx]) < BIG_READ_SIZE) || (iopb.size() < BIG_READ_SIZE)) && 
        (extra_bytes < READ_COALESCE_SIZE) && (buffer_used + len[idx] + extra_bytes <= buffer_size)) {
      // The space between the two reads is small enough we can coalesce.

      // We enforce that the current read or the current iopb must be small.
      // This is so we can "perfectly pack" buffers consisting of only big
      // reads - in such a case, read coalescing doesn't help much.
      m_idx_to_iopb.push_back(iopb_offset);
      m_idx_to_iopb_offset.push_back(pos[idx]-iopb.offset());
      iopb.set_size(pos[idx]+len[idx] - iopb.offset());
      buffer_used += (len[idx] + extra_bytes);
      m_extra_bytes += extra_bytes;
      continue;
    }
    // There is a big jump, but still space left in the temporary buffer.
    // Record our current iopb:
    m_iov.push_back(iopb);

    // Reset iopb
    iopb.set_offset(pos[idx]);
    iopb.set_data(buf + buffer_used);
    iopb.set_size(len[idx]);

    // Record location of this chunk.
    iopb_offset ++;

    m_idx_to_iopb.push_back(iopb_offset);
    m_idx_to_iopb_offset.push_back(0);

    buffer_used += len[idx];
  }
  m_iov.push_back(iopb);

  m_buffer_used += buffer_used;
  return idx;
}

/**
 * Unpack the optimized set of reads from the storage system and copy the
 * results in the order ROOT requested.
 */
void
ReadRepacker::unpack(char *buf)
{

  char * root_result_ptr = buf;
  int nbuf = m_idx_to_iopb.size();
  for (int idx=0; idx < nbuf; idx++) {
    int iov_idx = m_idx_to_iopb[idx];
    IOPosBuffer &iopb = m_iov[iov_idx];
    int iopb_offset = m_idx_to_iopb_offset[idx];
    char * io_result_ptr = static_cast<char *>(iopb.data()) + iopb_offset;
    // Note that we use the input buffer as a temporary where possible.
    // Hence, the source and destination can overlap; use memmove instead of memcpy.
    memmove(root_result_ptr, io_result_ptr, m_len[idx]);

    root_result_ptr += m_len[idx];
  }

}

void
ReadRepacker::reset(unsigned int nbuf)
{
  m_extra_bytes = 0;
  m_buffer_used = 0;

  // Number of buffers to storage typically decreases, but nbuf/2 is just an
  // somewhat-informed guess.
  m_iov.reserve(nbuf/2);
  m_iov.clear();
  m_idx_to_iopb.reserve(nbuf);
  m_idx_to_iopb.clear();
  m_idx_to_iopb_offset.reserve(nbuf);
  m_idx_to_iopb_offset.clear();
}

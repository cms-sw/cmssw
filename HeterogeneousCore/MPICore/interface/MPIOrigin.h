#ifndef HeterogeneousCore_MPICore_interface_MPIOrigin_h
#define HeterogeneousCore_MPICore_interface_MPIOrigin_h

#include "DataFormats/Common/interface/traits.h"

/* MPIOrigin
 *
 * Include the information that describe form which MPI process an Event originates from,
 * so that data products can be sent back to it.
 *
 * Data members are `int` to match types expected by MPI.
 *
 * This class is always transient, as it doesn't make sense to store it and read it from
 * a different, subsequent process.
 */

class MPIOrigin : public edm::DoNotRecordParents {
public:
  MPIOrigin() = default;
  MPIOrigin(int process, uint32_t stream) : process_(process), stream_(static_cast<int>(stream)) {}

  int process() const { return process_; }  // rank of the original MPI process
  int rank() const { return process_; }     // alias for process()

  int stream() const { return stream_; }  // EDM stream id within the original process
  int tag() const { return stream_; }     // alias for stream()

  bool valid() const { return (-1 != process_ and -1 != stream_); }

private:
  int process_ = -1;  // the rank of the MPI process the Even originates from
  int stream_ = -1;   // the EDM stream ID within the original process
};

#endif  // HeterogeneousCore_MPICore_interface_MPIOrigin_h

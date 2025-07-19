#ifndef HeterogeneousCore_MPICore_MPIToken_h
#define HeterogeneousCore_MPICore_MPIToken_h

#include <memory>

// forward declaration
class MPIChannel;

class MPIToken {
public:
  // default constructor, needed to write the type's dictionary
  MPIToken() = default;

  // user-defined constructor
  explicit MPIToken(std::shared_ptr<MPIChannel> channel) : channel_(channel) {}

  // access the data member
  MPIChannel* channel() const { return channel_.get(); }

private:
  // wrap the MPI communicator and destination
  std::shared_ptr<MPIChannel> channel_;
};

#endif  // HeterogeneousCore_MPICore_MPIToken_h

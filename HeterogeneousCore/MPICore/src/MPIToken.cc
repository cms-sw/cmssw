#include "HeterogeneousCore/MPICore/interface/MPIChannel.h"
#include "HeterogeneousCore/MPICore/interface/MPIToken.h"

// Store the MPIChannel in a shared_ptr so that copies of the MPIToken can refer to the same MPIChannel.
// When the last MPIToken is destroyed, the shared_ptr will call MPIChannel::sync() insyead of deleting it.
MPIToken::MPIToken(MPIChannel& channel) : channel_(&channel, [](MPIChannel* ptr) { ptr->sync(); }) {
  channel_->acquire();
}

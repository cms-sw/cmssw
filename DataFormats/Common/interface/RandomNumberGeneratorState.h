#ifndef DataFormats_Common_RandomNumberGeneratorState_h
#define DataFormats_Common_RandomNumberGeneratorState_h

/*----------------------------------------------------------------------
  
RandomNumberGeneratorState is used to communicate with an external process
----------------------------------------------------------------------*/

#include <vector>
namespace edm {
  struct RandomNumberGeneratorState {
    RandomNumberGeneratorState() = default;
    RandomNumberGeneratorState(std::vector<unsigned long> iState, long iSeed)
        : state_(std::move(iState)), seed_{iSeed} {}

    RandomNumberGeneratorState(RandomNumberGeneratorState const&) = default;
    RandomNumberGeneratorState(RandomNumberGeneratorState&&) = default;

    RandomNumberGeneratorState& operator=(RandomNumberGeneratorState const&) = default;
    RandomNumberGeneratorState& operator=(RandomNumberGeneratorState&&) = default;

    std::vector<unsigned long> state_;
    long seed_;
  };
}  // namespace edm
#endif

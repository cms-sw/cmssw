#include "HeterogeneousCore/SonicCore/interface/SonicRetryActionBase.h"
#include "HeterogeneousCore/SonicCore/interface/SonicClientBase.h"

class SonicRetrySameServerAction : public SonicRetryActionBase {
public:
  SonicRetrySameServerAction(const edm::ParameterSet& pset, SonicClientBase& client)
      : SonicRetryActionBase(pset, client), allowedTries_(pset.getUntrackedParameter<unsigned>("allowedTries", 0)) {}

  void start() override { tries_ = 0; };

protected:
  void retry() override;

private:
  unsigned allowedTries_, tries_;
};

void SonicRetrySameServerAction::retry() {
  ++tries_;
  //if max retries has not been exceeded, call evaluate again
  if (tries_ >= allowedTries_) {
    shouldRetry_ = false;  // Flip flag when max retries are reached
    edm::LogInfo("SonicRetrySameServerAction") << "Max retry attempts reached. No further retries.";
  }
  eval();
}

DEFINE_RETRY_ACTION(SonicRetrySameServerAction)

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonException.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonService.h"

TritonException::TritonException(std::string const& aCategory, bool signal)
    : cms::Exception(aCategory), signal_(signal) {
  if (signal_) {
    edm::Service<TritonService> ts;
    ts->notifyCallStatus(false);
  }
}

void TritonException::convertToWarning() const {
  //undo the previous signal
  if (signal_) {
    edm::Service<TritonService> ts;
    ts->notifyCallStatus(true);
  }

  edm::LogWarning(category()) << explainSelf();
}

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonException.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonService.h"

TritonException::TritonException(std::string const& aCategory, const TritonService* service)
    : cms::Exception(aCategory) {
  if (service) {
    service->notifyCallStatus(false);
  }
}

void TritonException::convertToWarning() const { edm::LogWarning(category()) << explainSelf(); }

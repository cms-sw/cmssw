#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonException.h"

TritonException::TritonException(std::string const& aCategory) : cms::Exception(aCategory) {}

void TritonException::convertToWarning() const { edm::LogWarning(category()) << explainSelf(); }

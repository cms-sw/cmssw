#include "HeterogeneousCore/SonicCore/interface/SonicRetryActionBase.h"

// Constructor implementation
SonicRetryActionBase::SonicRetryActionBase(const edm::ParameterSet& conf, SonicClientBase* client)
    : client_(client), shouldRetry_(true) {
  if (client_ == nullptr) {
    throw cms::Exception("SonicRetryActionBase") << "client pointer cannot be null";
  }
}

void SonicRetryActionBase::eval() { client_->evaluate(); }

void SonicRetryActionBase::finish(bool success) { client_->finish(success); }

EDM_REGISTER_PLUGINFACTORY(RetryActionFactory, "RetryActionFactory");

#include "HeterogeneousCore/SonicCore/interface/RetryActionBase.h"

// Constructor implementation
RetryActionBase::RetryActionBase(const edm::ParameterSet& conf, SonicClientBase* client)
    : client_(client), shouldRetry_(true) {
  if (client_ == nullptr) {
    throw cms::Exception("RetryActionBase") << "client pointer cannot be null";
  }
}

void RetryActionBase::eval() { client_->evaluate(); }

void RetryActionBase::finish(bool success) { client_->finish(success); }

EDM_REGISTER_PLUGINFACTORY(RetryActionFactory, "RetryActionFactory");

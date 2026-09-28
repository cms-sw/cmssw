#include "HeterogeneousCore/SonicCore/interface/SonicRetryActionBase.h"
#include "HeterogeneousCore/SonicCore/interface/SonicClientBase.h"

// Constructor implementation
SonicRetryActionBase::SonicRetryActionBase(const edm::ParameterSet& conf, SonicClientBase& client)
    : client_(client), shouldRetry_(true) {}

void SonicRetryActionBase::eval() { client_.evaluate(); }

void SonicRetryActionBase::finish(bool success) { client_.finish(success); }

EDM_REGISTER_PLUGINFACTORY(RetryActionFactory, "RetryActionFactory");

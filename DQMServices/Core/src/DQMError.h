#ifndef DQMSERVICES_CORE_DQMEXCEPTION_H
#define DQMSERVICES_CORE_DQMEXCEPTION_H

#include <stdexcept>
#if !WITHOUT_CMS_FRAMEWORK
#include "FWCore/Utilities/interface/EDMException.h"
using DQMError = cms::Exception;
#else
using DQMError = std::runtime_error;
#endif

void raiseDQMError(const char *context, const char *fmt, ...);
#endif

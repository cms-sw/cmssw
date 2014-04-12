#ifndef DQMSERVICES_CORE_DQMEXCEPTION_H
# define DQMSERVICES_CORE_DQMEXCEPTION_H

# include <stdexcept>
# if !WITHOUT_CMS_FRAMEWORK
#  include "FWCore/Utilities/interface/EDMException.h"
typedef cms::Exception DQMError;
# else
typedef std::runtime_error DQMError;
# endif

void raiseDQMError(const char *context, const char *fmt, ...);
#endif

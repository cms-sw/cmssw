#ifndef DQMSERVICES_CORE_DQM_DEFINITIONS_H
# define DQMSERVICES_CORE_DQM_DEFINITIONS_H

# include "DQMServices/Core/interface/DQMChannel.h"

namespace dqm
{
  /** Numeric constants for quality test results.  The smaller the
      number, the less severe the message.  */
  namespace qstatus
  {
    static const int OTHER		=  30;  //< Anything but 'ok','warning' or 'error'.
    static const int DISABLED		=  50;  //< Test has been disabled.
    static const int INVALID		=  60;  //< Problem preventing test from running.
    static const int INSUF_STAT		=  70;  //< Insufficient statistics.
    static const int DID_NOT_RUN	=  90;  //< Algorithm did not run.
    static const int STATUS_OK		=  100; //< Test was succesful.
    static const int WARNING		=  200; //< Test had some problems.
    static const int ERROR		=  300; //< Test has failed.
  }

  namespace me_util
  {
    typedef DQMChannel Channel;
  }
}

#endif // DQMSERVICES_CORE_DQM_DEFINITIONS_H

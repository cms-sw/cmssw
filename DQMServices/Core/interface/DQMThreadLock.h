#ifndef DQMSERVICES_CORE_DQM_THREAD_LOCK_H
# define DQMSERVICES_CORE_DQM_THREAD_LOCK_H

# include "FWCore/Framework/interface/Event.h"
# include "FWCore/ParameterSet/interface/ParameterSet.h"
# include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

/** Gateway to accessing DQM core in threads other than the CMSSW thread.  */
class DQMThreadLock
{
public:
  // Obtain access to DQM core from a thread other than the CMSSW thread.
  struct ExtraThread
  {
    ExtraThread(void);
    ~ExtraThread(void);
  };

  // Obtain access to DQM core in a service in the CMSSW thread.
  struct EDMService
  {
    EDMService(void);
    ~EDMService(void);
  };

  // For use by edm::Service, within DQMServices/Core only
  DQMThreadLock(const edm::ParameterSet &pset, edm::ActivityRegistry &ar);
  ~DQMThreadLock(void);
};

#endif // DQMSERVICES_CORE_DQM_THREAD_LOCK_H

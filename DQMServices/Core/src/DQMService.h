#ifndef DQMSERVICES_CORE_DQM_SERVICE_H
# define DQMSERVICES_CORE_DQM_SERVICE_H

# include "FWCore/Framework/interface/Event.h"
# include "FWCore/ParameterSet/interface/ParameterSet.h"
# include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

class DQMStore;
class DQMBasicNet;
namespace lat { class Regexp; }

/** A bridge to udpate the DQM network layer at the end of every event.  */
class DQMService
{
public:
  DQMService(const edm::ParameterSet &pset, edm::ActivityRegistry &ar);
  ~DQMService(void);
   
private:
  void flush(const edm::Event &, const edm::EventSetup &);
  void shutdown(void);

  DQMStore	*store_;
  DQMBasicNet	*net_;
  lat::Regexp	*filter_;
  double	lastFlush_;
  double	publishFrequency_;
};

#endif // DQMSERVICES_CORE_DQM_SERVICE_H

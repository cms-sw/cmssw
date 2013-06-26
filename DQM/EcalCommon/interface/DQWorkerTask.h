#ifndef DQWorkerTask_H
#define DQWorkerTask_H

#include <string>

#include "DQM/EcalCommon/interface/DQWorker.h"
#include "DQM/EcalCommon/interface/Collections.h"

#include "FWCore/Common/interface/TriggerResultsByName.h"

namespace ecaldqm {
  class DQWorkerTask : public DQWorker {
  public:
    DQWorkerTask(const edm::ParameterSet&, const edm::ParameterSet&, std::string const&);
    virtual ~DQWorkerTask() {}

    virtual void beginEvent(const edm::Event &, const edm::EventSetup &) {}
    virtual void endEvent(const edm::Event &, const edm::EventSetup &) {}

    virtual bool runsOn(unsigned);
    virtual const std::vector<std::pair<Collections, Collections> >& getDependencies();
    virtual bool filterRunType(const std::vector<short>&);
    virtual bool filterTrigger(const edm::TriggerResultsByName &);

    virtual void analyze(const void*, Collections){}

  protected:
    uint32_t collectionMask_;
    // list of dependencies between collections
    // first element depends on the second
    std::vector<std::pair<Collections, Collections> > dependencies_;
  };

}
#endif

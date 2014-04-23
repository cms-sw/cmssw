#ifndef DQWorkerClient_H
#define DQWorkerClient_H

#include <utility>

#include "DQM/EcalCommon/interface/DQWorker.h"

class DetId;

namespace ecaldqm
{
  class StatusManager;

  class DQWorkerClient : public DQWorker {
  public:
    enum ProcessType {
      kLumi,
      kRun,
      nProcessType
    };

    DQWorkerClient();
    virtual ~DQWorkerClient() {}

    static void fillDescriptions(edm::ParameterSetDescription&);

    void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

    void bookMEs(DQMStore&) override;
    void releaseMEs() override;

    void releaseSource();
    bool retrieveSource(DQMStore const&, ProcessType);

    void resetMEs();
    virtual void producePlots(ProcessType) = 0;

    void setStatusManager(StatusManager const& _manager) { statusManager_ = &_manager; }

    enum Quality {
      kBad = 0,
      kGood = 1,
      kUnknown = 2,
      kMBad = 3,
      kMGood = 4,
      kMUnknown = 5
    };

  protected:
    void setME(edm::ParameterSet const& _ps) final { DQWorker::setME(_ps); }
    void setSource(edm::ParameterSet const&) override;

    bool using_(std::string const& _name, ProcessType _type = kRun) const
    {
      MESetCollection::const_iterator itr(sources_.find(_name));
      if(itr == sources_.end()) return false;
      if(_type == kRun) return true;
      else return itr->second->getLumiFlag();
    }

    void towerAverage_(MESet&, MESet const&, float);

    MESetCollection sources_;
    std::set<std::string> qualitySummaries_;

    StatusManager const* statusManager_;
  };
}
#endif

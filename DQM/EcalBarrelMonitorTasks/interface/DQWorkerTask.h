#ifndef DQWorkerTask_H
#define DQWorkerTask_H

#include <string>

#include "DQM/EcalCommon/interface/DQWorker.h"
#include "Collections.h"

#include "DataFormats/EcalRawData/interface/EcalDCCHeaderBlock.h"

#include "FWCore/Common/interface/TriggerResultsByName.h"

namespace ecaldqm {

  struct Dependency {
    Collections dependant;
    std::set<Collections> requisite;

    Dependency() : dependant(Collections(-1)), requisite() {}
    Dependency(Collections _d, int _r1 = -1, int _r2 = -1, int _r3 = -1, int _r4 = -1) :
      dependant(_d),
      requisite()
    {
      if(_r1 >= 0) append(Collections(_r1));
      if(_r2 >= 0) append(Collections(_r2));
      if(_r3 >= 0) append(Collections(_r3));
      if(_r4 >= 0) append(Collections(_r4));
    }
    void append(Collections _r)
    {
      if(_r != int(dependant)) requisite.insert(_r);
    }
    void append(std::set<Collections> const& _s)
    {
      for(std::set<Collections>::const_iterator sItr(_s.begin()); sItr != _s.end(); ++sItr)
        append(*sItr);
    }
  };

  struct DependencySet {
    DependencySet() :
      set_()
    {}
    void push_back(Dependency const& _d)
    {
      std::vector<Dependency>::iterator dItr(set_.begin());
      std::vector<Dependency>::iterator dEnd(set_.end());
      for(; dItr != dEnd; ++dItr)
        if(dItr->dependant == _d.dependant) dItr->append(_d.requisite);
      if(dItr == dEnd) set_.push_back(_d);
    }
    std::vector<Collections> formSequence() const
    {
      std::vector<Collections> sequence;
      for(unsigned iD(0); iD < set_.size(); iD++){
        if(std::find(sequence.begin(), sequence.end(), set_[iD].dependant) != sequence.end()) continue;
        formSequenceFragment_(set_[iD], sequence, sequence.end());
      }
      return sequence;
    }

    private:
    std::vector<Dependency> set_;

    void formSequenceFragment_(Dependency const&, std::vector<Collections>&, std::vector<Collections>::iterator) const;
  };

  class DQWorkerTask : public DQWorker {
  public:
    typedef EcalDCCHeaderBlock::EcalDCCEventSettings EventSettings;

    DQWorkerTask(edm::ParameterSet const&, edm::ParameterSet const&, std::string const&);
    virtual ~DQWorkerTask() {}

    virtual void beginEvent(const edm::Event &, const edm::EventSetup &) {}
    virtual void endEvent(const edm::Event &, const edm::EventSetup &) {}

    virtual bool filterRunType(const std::vector<short>&);
    virtual bool filterTrigger(const edm::TriggerResultsByName &);
    bool runsOn(unsigned);
    virtual void setDependencies(DependencySet&) {}

    virtual void analyze(const void*, Collections){}

    void softReset();
    void recoverStats();

  protected:
    std::vector<bool> collectionMask_;
    std::set<std::string> resettable_;
  };

  inline
  bool
  DQWorkerTask::runsOn(unsigned _collection)
  {
    if(_collection >= nProcessedObjects) return false;
    return collectionMask_[_collection];
  }
}
#endif

#ifndef DQWorkerTask_H
#define DQWorkerTask_H

#include "DQM/EcalCommon/interface/DQWorker.h"

#include "DataFormats/EcalRawData/interface/EcalDCCHeaderBlock.h"

#include "Collections.h"

#include <bitset>

namespace edm
{
  class TriggerResultsByName;
  class ConsumesCollector;
}

namespace ecaldqm
{

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

    DQWorkerTask();
    virtual ~DQWorkerTask() {}

    static void fillDescriptions(edm::ParameterSetDescription&);

    virtual void beginEvent(edm::Event const&, edm::EventSetup const&) {}
    virtual void endEvent(edm::Event const&, edm::EventSetup const&) {}

    virtual bool filterRunType(short const*) { return true; };
    virtual bool filterTrigger(edm::TriggerResultsByName const&) { return true; };
    virtual void addDependencies(DependencySet&) {}

    // mechanisms to register EDGetTokens for any additional objects used internally
    virtual void setTokens(edm::ConsumesCollector&) {}

    // "Dry-run" mode when passed a null pointer
    // Returns true if the module runs on the collection
    virtual bool analyze(void const*, Collections) { return false; }

    void softReset();
    void recoverStats();

  protected:
    void setME(edm::ParameterSet const&) final;

    std::set<std::string> resettable_;
  };
}
#endif

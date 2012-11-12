#ifndef TrigPrimTask_H
#define TrigPrimTask_H

#include "DQWorkerTask.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

namespace ecaldqm {

  class TrigPrimTask : public DQWorkerTask {
  public:
    TrigPrimTask(edm::ParameterSet const&, edm::ParameterSet const&);
    ~TrigPrimTask() {}

    void setDependencies(DependencySet&);

    void bookMEs();

    void analyze(const void*, Collections);

    void beginEvent(const edm::Event &, const edm::EventSetup &);

    void runOnRealTPs(const EcalTrigPrimDigiCollection &);
    void runOnEmulTPs(const EcalTrigPrimDigiCollection &);
    void runOnDigis(const EcalDigiCollection &);

    enum Constants {
      nBXBins = 15
    };

  private:
    EcalTrigPrimDigiCollection const* realTps_;

    bool runOnEmul_;

/*     std::string HLTCaloPath_; */
/*     std::string HLTMuonPath_; */
/*     bool HLTCaloBit_; */
/*     bool HLTMuonBit_; */

    int bxBinEdges_[nBXBins + 1];
    double bxBin_;

    std::map<uint32_t, unsigned> towerReadouts_;
  };

  inline void TrigPrimTask::analyze(const void* _p, Collections _collection){
    switch(_collection){
    case kTrigPrimDigi:
      runOnRealTPs(*static_cast<const EcalTrigPrimDigiCollection*>(_p));
      break;
    case kTrigPrimEmulDigi:
      runOnEmulTPs(*static_cast<const EcalTrigPrimDigiCollection*>(_p));
      break;
    case kEBDigi:
    case kEEDigi:
      runOnDigis(*static_cast<const EcalDigiCollection*>(_p));
      break;
    default:
      break;
    }
  }

}

#endif


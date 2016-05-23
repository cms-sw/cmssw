#ifndef TrigPrimTask_H
#define TrigPrimTask_H

#include "DQWorkerTask.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/EcalObjects/interface/EcalTPGTowerStatus.h"
#include "CondFormats/EcalObjects/interface/EcalTPGStripStatus.h"

namespace ecaldqm {

  class TrigPrimTask : public DQWorkerTask {
  public:
    TrigPrimTask();
    ~TrigPrimTask() {}

    void addDependencies(DependencySet&) override;

    bool analyze(void const*, Collections) override;

    void beginRun(edm::Run const&, edm::EventSetup const&) override;
    void beginEvent(edm::Event const&, edm::EventSetup const&) override;

    void runOnRealTPs(EcalTrigPrimDigiCollection const&);
    void runOnEmulTPs(EcalTrigPrimDigiCollection const&);
    template<typename DigiCollection> void runOnDigis(DigiCollection const&);

    enum Constants {
      nBXBins = 15
    };

  private:
    void setParams(edm::ParameterSet const&) override;

    EcalTrigPrimDigiCollection const* realTps_;

    bool runOnEmul_;

/*     std::string HLTCaloPath_; */
/*     std::string HLTMuonPath_; */
/*     bool HLTCaloBit_; */
/*     bool HLTMuonBit_; */

    int bxBinEdges_[nBXBins + 1];
    double bxBin_;

    std::map<uint32_t, unsigned> towerReadouts_;

    edm::ESHandle<EcalTPGTowerStatus> TTStatusRcd;
    edm::ESHandle<EcalTPGStripStatus> StripStatusRcd;

  };

  inline bool TrigPrimTask::analyze(void const* _p, Collections _collection){
    switch(_collection){
    case kTrigPrimDigi:
      if(_p) runOnRealTPs(*static_cast<EcalTrigPrimDigiCollection const*>(_p));
      return true;
      break;
    case kTrigPrimEmulDigi:
      if(_p && runOnEmul_) runOnEmulTPs(*static_cast<EcalTrigPrimDigiCollection const*>(_p));
      return runOnEmul_;
      break;
    case kEBDigi:
      if(_p) runOnDigis(*static_cast<EBDigiCollection const*>(_p));
      return true;
      break;
    case kEEDigi:
      if(_p) runOnDigis(*static_cast<EEDigiCollection const*>(_p));
      return true;
      break;
    default:
      break;
    }
    return false;
  }

}

#endif


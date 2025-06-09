#ifndef TrigPrimTask_H
#define TrigPrimTask_H

#include "DQWorkerTask.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/TCDS/interface/TCDSRecord.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "CondFormats/EcalObjects/interface/EcalTPGTowerStatus.h"
#include "CondFormats/EcalObjects/interface/EcalTPGStripStatus.h"
#include "CondFormats/DataRecord/interface/EcalTPGTowerStatusRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGStripStatusRcd.h"

#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"

namespace ecaldqm {

  class TrigPrimTask : public DQWorkerTask {
  public:
    TrigPrimTask();
    ~TrigPrimTask() override {}

    void addDependencies(DependencySet&) override;

    void beginRun(edm::Run const&, edm::EventSetup const&) override;
    void beginEvent(edm::Event const&, edm::EventSetup const&, bool const&, bool&) override;

    bool analyze(void const*, Collections) override;

    void runOnRealTPs(EcalTrigPrimDigiCollection const&);
    void runOnEmulTPs(EcalTrigPrimDigiCollection const&);
    template <typename DigiCollection>
    void runOnDigis(DigiCollection const&);
    void runOnRecHits(EcalRecHitCollection const&, Collections); 
    
    void setTokens(edm::ConsumesCollector&) override;

    enum Constants { nBXBins = 15 };

    void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

  private:
    void setParams(edm::ParameterSet const&) override;

    EcalTrigPrimDigiCollection const* realTps_;

    bool runOnEmul_;

    /*     std::string HLTCaloPath_; */
    /*     std::string HLTMuonPath_; */
    /*     bool HLTCaloBit_; */
    /*     bool HLTMuonBit_; */

    std::vector<int> bxBinEdges_;
    std::vector<int> bxBinEdgesFine_;
    double bxBin_;
    double bxBinFine_;

    double etSum_;
    double etSpikeMatchSum_;

    std::map<uint32_t, unsigned> towerReadouts_;

    edm::ESGetToken<EcalTPGTowerStatus, EcalTPGTowerStatusRcd> TTStatusRcd_;
    edm::ESGetToken<EcalTPGStripStatus, EcalTPGStripStatusRcd> StripStatusRcd_;
    const EcalTPGTowerStatus* TTStatus;
    const EcalTPGStripStatus* StripStatus;

    edm::InputTag lhcStatusInfoCollectionTag_;
    edm::EDGetTokenT<TCDSRecord> lhcStatusInfoRecordToken_;
  
    std::map<EcalTrigTowerDetId, float> mapTowerMaxRecHitEnergy_;
    std::map<EcalTrigTowerDetId, int> mapTowerOfflineSpikes_;
    edm::ESGetToken<EcalSeverityLevelAlgo, EcalSeverityLevelAlgoRcd> severityToken_;
    const EcalSeverityLevelAlgo* sevLevel;
  };

  inline bool TrigPrimTask::analyze(void const* _p, Collections _collection) {
    switch (_collection) {
      case kEBRecHit:
      case kEERecHit:
        if (_p)
          runOnRecHits(*static_cast<EcalRecHitCollection const*>(_p), _collection);
        return true;
        break;
      case kTrigPrimDigi:
        if (_p)
          runOnRealTPs(*static_cast<EcalTrigPrimDigiCollection const*>(_p));
        return true;
        break;
      case kTrigPrimEmulDigi:
        if (_p && runOnEmul_)
          runOnEmulTPs(*static_cast<EcalTrigPrimDigiCollection const*>(_p));
        return runOnEmul_;
        break;
      case kEBDigi:
        if (_p)
          runOnDigis(*static_cast<EBDigiCollection const*>(_p));
        return true;
        break;
      case kEEDigi:
        if (_p)
          runOnDigis(*static_cast<EEDigiCollection const*>(_p));
        return true;
        break;
     default:
        break;
    }
    return false;
  }

}  // namespace ecaldqm

#endif

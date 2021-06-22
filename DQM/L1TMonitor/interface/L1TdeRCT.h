#ifndef L1TdeRCT_H
#define L1TdeRCT_H

// system include files
#include <memory>
#include <unistd.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <bitset>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// DQM
#include "DQMServices/Core/interface/DQMStore.h"

// GCT and RCT data formats
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
// TPGs
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

#include "DQMServices/Core/interface/DQMOneEDAnalyzer.h"

class RunInfoRcd;
class RunInfo;
// Trigger Headers
//
// class declaration
//
namespace l1tderct {
  struct Empty {};
}  // namespace l1tderct

class L1TdeRCT : public DQMOneEDAnalyzer<edm::LuminosityBlockCache<l1tderct::Empty>> {
public:
  // Constructor
  L1TdeRCT(const edm::ParameterSet& ps);

  // Destructor
  ~L1TdeRCT() override;

protected:
  // Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  //For FED vector monitoring
  void bookHistograms(DQMStore::IBooker& ibooker, const edm::Run&, const edm::EventSetup&) override;
  std::shared_ptr<l1tderct::Empty> globalBeginLuminosityBlock(const edm::LuminosityBlock&,
                                                              const edm::EventSetup&) const override;
  void globalEndLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) final {}
  void readFEDVector(MonitorElement*, const edm::EventSetup&, const bool isLumitransition = true) const;

private:
  // ----------member data ---------------------------

  // begin GT decision information
  MonitorElement* triggerAlgoNumbers_;

  // trigger type information
  MonitorElement* triggerType_;

  // begin region information
  MonitorElement* rctRegDataOcc1D_;
  MonitorElement* rctRegEmulOcc1D_;
  MonitorElement* rctRegMatchedOcc1D_;
  MonitorElement* rctRegUnmatchedDataOcc1D_;
  MonitorElement* rctRegUnmatchedEmulOcc1D_;
  MonitorElement* rctRegSpEffOcc1D_;
  MonitorElement* rctRegSpIneffOcc1D_;

  MonitorElement* rctRegEff1D_;
  MonitorElement* rctRegIneff1D_;
  MonitorElement* rctRegOvereff1D_;
  MonitorElement* rctRegSpEff1D_;
  MonitorElement* rctRegSpIneff1D_;

  MonitorElement* rctRegDataOcc2D_;
  MonitorElement* rctRegEmulOcc2D_;
  MonitorElement* rctRegMatchedOcc2D_;
  MonitorElement* rctRegUnmatchedDataOcc2D_;
  MonitorElement* rctRegUnmatchedEmulOcc2D_;
  //  MonitorElement *rctRegDeltaEt2D_;
  MonitorElement* rctRegSpEffOcc2D_;
  MonitorElement* rctRegSpIneffOcc2D_;

  MonitorElement* rctRegEff2D_;
  MonitorElement* rctRegIneff2D_;
  MonitorElement* rctRegOvereff2D_;
  MonitorElement* rctRegSpEff2D_;
  MonitorElement* rctRegSpIneff2D_;

  MonitorElement* rctRegBitOn_;
  MonitorElement* rctRegBitOff_;
  MonitorElement* rctRegBitDiff_;

  // end region information

  // begin bit information
  MonitorElement* rctBitEmulOverFlow2D_;
  MonitorElement* rctBitDataOverFlow2D_;
  MonitorElement* rctBitMatchedOverFlow2D_;
  MonitorElement* rctBitUnmatchedEmulOverFlow2D_;
  MonitorElement* rctBitUnmatchedDataOverFlow2D_;
  MonitorElement* rctBitOverFlowEff2D_;
  MonitorElement* rctBitOverFlowIneff2D_;
  MonitorElement* rctBitOverFlowOvereff2D_;
  MonitorElement* rctBitEmulTauVeto2D_;
  MonitorElement* rctBitDataTauVeto2D_;
  MonitorElement* rctBitMatchedTauVeto2D_;
  MonitorElement* rctBitUnmatchedEmulTauVeto2D_;
  MonitorElement* rctBitUnmatchedDataTauVeto2D_;
  MonitorElement* rctBitTauVetoEff2D_;
  MonitorElement* rctBitTauVetoIneff2D_;
  MonitorElement* rctBitTauVetoOvereff2D_;
  MonitorElement* rctBitEmulMip2D_;
  MonitorElement* rctBitDataMip2D_;
  MonitorElement* rctBitMatchedMip2D_;
  MonitorElement* rctBitUnmatchedEmulMip2D_;
  MonitorElement* rctBitUnmatchedDataMip2D_;
  MonitorElement* rctBitMipEff2D_;
  MonitorElement* rctBitMipIneff2D_;
  MonitorElement* rctBitMipOvereff2D_;
  MonitorElement* rctBitEmulQuiet2D_;
  MonitorElement* rctBitDataQuiet2D_;
  MonitorElement* rctBitMatchedQuiet2D_;
  MonitorElement* rctBitUnmatchedEmulQuiet2D_;
  MonitorElement* rctBitUnmatchedDataQuiet2D_;
  // QUIETBIT: To add quiet bit information, uncomment following 3 lines:
  // MonitorElement *rctBitQuietEff2D_;
  // MonitorElement *rctBitQuietIneff2D_;
  // MonitorElement *rctBitQuietOvereff2D_;
  MonitorElement* rctBitEmulHfPlusTau2D_;
  MonitorElement* rctBitDataHfPlusTau2D_;
  MonitorElement* rctBitMatchedHfPlusTau2D_;
  MonitorElement* rctBitUnmatchedEmulHfPlusTau2D_;
  MonitorElement* rctBitUnmatchedDataHfPlusTau2D_;
  MonitorElement* rctBitHfPlusTauEff2D_;
  MonitorElement* rctBitHfPlusTauIneff2D_;
  MonitorElement* rctBitHfPlusTauOvereff2D_;

  // end bit information

  MonitorElement* rctInputTPGEcalOcc_;
  MonitorElement* rctInputTPGEcalOccNoCut_;
  MonitorElement* rctInputTPGEcalRank_;
  MonitorElement* rctInputTPGHcalOcc_;
  MonitorElement* rctInputTPGHcalRank_;
  MonitorElement* rctInputTPGHcalSample_;

  MonitorElement* rctIsoEmDataOcc_;
  MonitorElement* rctIsoEmEmulOcc_;
  MonitorElement* rctIsoEmEff1Occ_;
  MonitorElement* rctIsoEmEff2Occ_;
  MonitorElement* rctIsoEmIneff2Occ_;
  MonitorElement* rctIsoEmIneffOcc_;
  MonitorElement* rctIsoEmOvereffOcc_;
  MonitorElement* rctIsoEmEff1_;
  MonitorElement* rctIsoEmEff2_;
  MonitorElement* rctIsoEmIneff2_;
  MonitorElement* rctIsoEmIneff_;
  MonitorElement* rctIsoEmOvereff_;

  MonitorElement* rctIsoEmDataOcc1D_;
  MonitorElement* rctIsoEmEmulOcc1D_;
  MonitorElement* rctIsoEmEff1Occ1D_;
  MonitorElement* rctIsoEmEff2Occ1D_;
  MonitorElement* rctIsoEmIneff2Occ1D_;
  MonitorElement* rctIsoEmIneffOcc1D_;
  MonitorElement* rctIsoEmOvereffOcc1D_;
  MonitorElement* rctIsoEmEff1oneD_;
  MonitorElement* rctIsoEmEff2oneD_;
  MonitorElement* rctIsoEmIneff2oneD_;
  MonitorElement* rctIsoEmIneff1D_;
  MonitorElement* rctIsoEmOvereff1D_;

  MonitorElement* rctIsoEmBitOn_;
  MonitorElement* rctIsoEmBitOff_;
  MonitorElement* rctIsoEmBitDiff_;

  MonitorElement* rctNisoEmDataOcc_;
  MonitorElement* rctNisoEmEmulOcc_;
  MonitorElement* rctNisoEmEff1Occ_;
  MonitorElement* rctNisoEmEff2Occ_;
  MonitorElement* rctNisoEmIneff2Occ_;
  MonitorElement* rctNisoEmIneffOcc_;
  MonitorElement* rctNisoEmOvereffOcc_;
  MonitorElement* rctNisoEmEff1_;
  MonitorElement* rctNisoEmEff2_;
  MonitorElement* rctNisoEmIneff2_;
  MonitorElement* rctNisoEmIneff_;
  MonitorElement* rctNisoEmOvereff_;

  MonitorElement* rctNisoEmDataOcc1D_;
  MonitorElement* rctNisoEmEmulOcc1D_;
  MonitorElement* rctNisoEmEff1Occ1D_;
  MonitorElement* rctNisoEmEff2Occ1D_;
  MonitorElement* rctNisoEmIneff2Occ1D_;
  MonitorElement* rctNisoEmIneffOcc1D_;
  MonitorElement* rctNisoEmOvereffOcc1D_;
  MonitorElement* rctNisoEmEff1oneD_;
  MonitorElement* rctNisoEmEff2oneD_;
  MonitorElement* rctNisoEmIneff2oneD_;
  MonitorElement* rctNisoEmIneff1D_;
  MonitorElement* rctNisoEmOvereff1D_;

  MonitorElement* rctNIsoEmBitOn_;
  MonitorElement* rctNIsoEmBitOff_;
  MonitorElement* rctNIsoEmBitDiff_;

  MonitorElement* rctIsoEffChannel_[396];
  MonitorElement* rctIsoIneffChannel_[396];
  MonitorElement* rctIsoOvereffChannel_[396];

  MonitorElement* rctNisoEffChannel_[396];
  MonitorElement* rctNisoIneffChannel_[396];
  MonitorElement* rctNisoOvereffChannel_[396];

  // begin region channel information
  MonitorElement* rctRegEffChannel_[396];
  MonitorElement* rctRegIneffChannel_[396];
  MonitorElement* rctRegOvereffChannel_[396];

  //efficiency
  MonitorElement* trigEffThresh_;
  MonitorElement* trigEffThreshOcc_;
  MonitorElement* trigEffTriggThreshOcc_;
  MonitorElement* trigEff_[396];
  MonitorElement* trigEffOcc_[396];
  MonitorElement* trigEffTriggOcc_[396];

  // end region channel information

  //begin fed vector information
  static const int crateFED[108];
  MonitorElement* fedVectorMonitorRUN_;
  MonitorElement* fedVectorMonitorLS_;
  ///////////////////////////////

  int nev_;                 // Number of events processed
  std::string histFolder_;  // base dqm folder
  bool verbose_;
  bool singlechannelhistos_;

  edm::EDGetTokenT<L1CaloRegionCollection> rctSourceEmul_rgnEmul_;
  edm::EDGetTokenT<L1CaloEmCollection> rctSourceEmul_emEmul_;
  edm::EDGetTokenT<L1CaloRegionCollection> rctSourceData_rgnData_;
  edm::EDGetTokenT<L1CaloEmCollection> rctSourceData_emData_;
  edm::EDGetTokenT<L1CaloRegionCollection> gctSourceData_rgnData_;
  edm::EDGetTokenT<L1CaloEmCollection> gctSourceData_emData_;
  edm::EDGetTokenT<EcalTrigPrimDigiCollection> ecalTPGData_;
  edm::EDGetTokenT<HcalTrigPrimDigiCollection> hcalTPGData_;
  edm::EDGetTokenT<L1GlobalTriggerReadoutRecord> gtDigisLabel_;
  edm::ESGetToken<RunInfo, RunInfoRcd> runInfoToken_;
  edm::ESGetToken<RunInfo, RunInfoRcd> runInfolumiToken_;
  std::string gtEGAlgoName_;  // name of algo to determine EG trigger threshold
  int doubleThreshold_;       // value of ET at which to make 2-D eff plot

  /// filter TriggerType
  int filterTriggerType_;
  int selectBX_;

  std::string dataInputTagName_;

  int trigCount, notrigCount;

protected:
  void DivideME1D(MonitorElement* numerator, MonitorElement* denominator, MonitorElement* result);
  void DivideME2D(MonitorElement* numerator, MonitorElement* denominator, MonitorElement* result);
};

#endif

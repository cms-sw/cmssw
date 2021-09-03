// -*- C++ -*-
//
// Package:    Calibration/EcalCalibAlgos
// Class:      ECALpedestalPCLHarvester
//
/**\class ECALpedestalPCLHarvester ECALpedestalPCLHarvester.cc 

 Description: Fill DQM histograms with pedestals. Intended to be used on laser data from the TestEnablesEcalHcal dataset

 
*/
//
// Original Author:  Stefano Argiro
//         Created:  Wed, 22 Mar 2017 14:46:48 GMT
//
//

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"

class ECALpedestalPCLHarvester : public DQMEDHarvester {
public:
  explicit ECALpedestalPCLHarvester(const edm::ParameterSet& ps);
  void endRun(edm::Run const& run, edm::EventSetup const& isetup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void dqmEndJob(DQMStore::IBooker& ibooker_, DQMStore::IGetter& igetter_) override;

  void dqmPlots(const EcalPedestals& newpeds, DQMStore::IBooker& ibooker);

  bool checkVariation(const EcalPedestalsMap& oldPedestals, const EcalPedestalsMap& newPedestals);
  bool checkStatusCode(const DetId& id);
  bool isGood(const DetId& id);

  std::vector<int> chStatusToExclude_;
  const int minEntries_;

  int entriesEB_[EBDetId::kSizeForDenseIndexing];
  int entriesEE_[EEDetId::kSizeForDenseIndexing];
  const bool checkAnomalies_;        // whether or not to avoid creating sqlite file in case of many changed pedestals
  const double nSigma_;              // threshold in sigmas to define a pedestal as changed
  const double thresholdAnomalies_;  // threshold (fraction of changed pedestals) to avoid creation of sqlite file
  const std::string dqmDir_;         // DQM directory where histograms are stored
  const std::string labelG6G1_;      // DB label from which pedestals for G6 and G1 are to be copied
  const float threshDiffEB_;         // if the new pedestals differs more than this from old, keep old
  const float threshDiffEE_;         // same as above for EE. Stray channel protection
  const float threshChannelsAnalyzed_;  // threshold for minimum percentage of channels analized to produce DQM plots

  // ES token
  const edm::ESGetToken<EcalChannelStatus, EcalChannelStatusRcd> channelsStatusToken_;
  const edm::ESGetToken<EcalPedestals, EcalPedestalsRcd> pedestalsToken_;
  const edm::ESGetToken<EcalPedestals, EcalPedestalsRcd> g6g1PedestalsToken_;

  const EcalPedestals* currentPedestals_;
  const EcalPedestals* g6g1Pedestals_;
  const EcalChannelStatus* channelStatus_;
};

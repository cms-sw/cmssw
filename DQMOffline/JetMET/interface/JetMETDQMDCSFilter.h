#ifndef DQMOffline_JetMET_JetMETDQMDCSFilter_H
#define DQMOffline_JetMET_JetMETDQMDCSFilter_H

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/OnlineMetaData/interface/DCSRecord.h"
#include "DataFormats/Scalers/interface/DcsStatus.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EDConsumerBase.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

class JetMETDQMDCSFilter {
public:
  JetMETDQMDCSFilter(const edm::ParameterSet&, edm::ConsumesCollector&);
  JetMETDQMDCSFilter(const std::string& detectorTypes,
                     edm::ConsumesCollector&,
                     const bool verbose = false,
                     const bool alwaysPass = false);
  ~JetMETDQMDCSFilter();
  bool filter(const edm::Event& evt, const edm::EventSetup& es);
  bool passPIX, passSiStrip, passECAL, passHBHE, passHF, passHO, passES, passMuon;
  edm::EDGetTokenT<DcsStatusCollection> scalarsToken_;
  edm::EDGetTokenT<DCSRecord> dcsRecordToken_;

private:
  template <typename T>
  void checkDCSInfoPerPartition(const T& DCS);
  void initializeVars();
  bool verbose_;
  bool filter_;
  bool detectorOn_;
  std::string detectorTypes_;
  std::map<std::string, std::vector<int>> associationMap_;
  std::map<std::string, bool> passPerDet_;
};

#endif

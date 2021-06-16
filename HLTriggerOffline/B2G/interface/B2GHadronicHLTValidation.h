// -*- C++ -*-
//
// Package:    HLTriggerOffline/B2G
// Class:      B2GHadronicHLTValidation
//
/**\class B2GHadronicHLTValidation B2GHadronicHLTValidation.h
 HLTriggerOffline/B2G/interface/B2GHadronicHLTValidation.h

 Description: compute efficiencies of trigger paths on offline reco selection
 with respect to pt and eta

 Implementation:
     harvesting
*/
//
// Original Author:  Elvire Bouvier
//         Created:  Thu, 16 Jan 2014 16:27:35 GMT
//
//
#ifndef B2GHADRONICHLTVALIDATION
#define B2GHADRONICHLTVALIDATION

// system include files
#include <memory>

// user include files
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

//
// class declaration
//

class B2GHadronicHLTValidation : public DQMEDAnalyzer {
public:
  explicit B2GHadronicHLTValidation(const edm::ParameterSet &);
  ~B2GHadronicHLTValidation() override;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  /// deduce monitorPath from label, the label is expected
  /// to be of type 'selectionPath:monitorPath'
  std::string monitorPath(const std::string &label) const { return label.substr(label.find(':') + 1); };
  /// set configurable labels for trigger monitoring histograms
  void triggerBinLabels(const std::vector<std::string> &labels);

  // ----------member data ---------------------------
  // DQM
  std::string sDir_;
  MonitorElement *hNumJetPt;
  MonitorElement *hDenJetPt;
  MonitorElement *hNumJetEta;
  MonitorElement *hDenJetEta;
  MonitorElement *hNumTriggerMon;
  MonitorElement *hDenTriggerMon;
  // Jets
  edm::Ptr<reco::Jet> jet_;
  std::string sJets_;
  edm::EDGetTokenT<edm::View<reco::Jet>> tokJets_;
  double ptJets_;
  double ptJets0_;
  double ptJets1_;
  double etaJets_;
  unsigned int minJets_;
  double htMin_;
  // Trigger
  std::string sTrigger_;
  edm::EDGetTokenT<edm::TriggerResults> tokTrigger_;
  std::vector<std::string> vsPaths_;
  // Flags
  bool isAll_ = false;
  bool isSel_ = false;
};

inline void B2GHadronicHLTValidation::triggerBinLabels(const std::vector<std::string> &labels) {
  for (unsigned int idx = 0; idx < labels.size(); ++idx) {
    hNumTriggerMon->setBinLabel(idx + 1, "[" + monitorPath(labels[idx]) + "]", 1);
    hDenTriggerMon->setBinLabel(idx + 1, "[" + monitorPath(labels[idx]) + "]", 1);
  }
}

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
inline B2GHadronicHLTValidation::B2GHadronicHLTValidation(const edm::ParameterSet &iConfig)
    : sDir_(iConfig.getUntrackedParameter<std::string>("sDir", "HLTValidation/B2G/Efficiencies/")),
      sJets_(iConfig.getUntrackedParameter<std::string>("sJets", "ak5PFJets")),
      ptJets_(iConfig.getUntrackedParameter<double>("ptJets", 0.)),
      ptJets0_(iConfig.getUntrackedParameter<double>("ptJets0", 0.)),
      ptJets1_(iConfig.getUntrackedParameter<double>("ptJets1", 0.)),
      etaJets_(iConfig.getUntrackedParameter<double>("etaJets", 0.)),
      minJets_(iConfig.getUntrackedParameter<unsigned int>("minJets", 0)),
      htMin_(iConfig.getUntrackedParameter<double>("htMin", 0.0)),
      sTrigger_(iConfig.getUntrackedParameter<std::string>("sTrigger", "TriggerResults")),
      vsPaths_(iConfig.getUntrackedParameter<std::vector<std::string>>("vsPaths"))

{
  // Jets
  tokJets_ = consumes<edm::View<reco::Jet>>(edm::InputTag(sJets_));
  // Trigger
  tokTrigger_ = consumes<edm::TriggerResults>(edm::InputTag(sTrigger_, "", "HLT"));
}

inline B2GHadronicHLTValidation::~B2GHadronicHLTValidation() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}
#endif

// define this as a plug-in
DEFINE_FWK_MODULE(B2GHadronicHLTValidation);

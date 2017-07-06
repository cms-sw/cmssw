#ifndef LepHTMonitor_H
#define LepHTMonitor_H

//event
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

//DQM
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

//Electron
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

//Muon
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"

//MET
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METCollection.h"

//Jets
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/BTauReco/interface/JetTag.h"

//Trigger
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

//Vertices
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

//Conversions
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"

//Beam spot
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

class GenericTriggerEventFlag;

class LepHTMonitor: public DQMEDAnalyzer{

public:
  LepHTMonitor(const edm::ParameterSet& ps);
  virtual ~LepHTMonitor();

protected:
  void dqmBeginRun(const edm::Run &run, const edm::EventSetup &e) override;
  void bookHistograms(DQMStore::IBooker &ibooker, const edm::Run &, const edm::EventSetup &) override;
  void beginLuminosityBlock(const edm::LuminosityBlock &lumi, const edm::EventSetup &eSetup)  override;
  void analyze(const edm::Event &e, const edm::EventSetup &eSetup) override;
  void endLuminosityBlock(const edm::LuminosityBlock &lumi, const edm::EventSetup &eSetup) override;
  void endRun(const edm::Run &run, const edm::EventSetup &eSetup) override;

private:
  //variables from config file
  edm::InputTag theElectronTag_;
  edm::EDGetTokenT<reco::GsfElectronCollection> theElectronCollection_;
  edm::InputTag theMuonTag_;
  edm::EDGetTokenT<reco::MuonCollection> theMuonCollection_;
  edm::InputTag thePfMETTag_;
  edm::EDGetTokenT<reco::PFMETCollection> thePfMETCollection_;
  edm::InputTag thePfJetTag_;
  edm::EDGetTokenT<reco::PFJetCollection> thePfJetCollection_;
  edm::InputTag theJetTagTag_;
  edm::EDGetTokenT<reco::JetTagCollection> theJetTagCollection_;

  edm::InputTag theVertexCollectionTag_;
  edm::EDGetTokenT<reco::VertexCollection> theVertexCollection_;
  edm::InputTag theConversionCollectionTag_;
  edm::EDGetTokenT<reco::ConversionCollection> theConversionCollection_;
  edm::InputTag theBeamSpotTag_;
  edm::EDGetTokenT<reco::BeamSpot> theBeamSpot_;
  GenericTriggerEventFlag* num_genTriggerEventFlag_;
  GenericTriggerEventFlag* den_lep_genTriggerEventFlag_;
  GenericTriggerEventFlag* den_HT_genTriggerEventFlag_;

  std::string triggerPath_;

  double jetPtCut_;
  double jetEtaCut_;
  double metCut_;
  double htCut_;
  double nmusCut_;
  double nelsCut_;
  double lep_pt_threshold_;
  double ht_threshold_;
  double met_threshold_;

  // Histograms

  MonitorElement* h_leptonTurnOn_num_;
  MonitorElement* h_leptonTurnOn_den_;
  MonitorElement* h_lepEtaTurnOn_num_;
  MonitorElement* h_lepEtaTurnOn_den_;
  MonitorElement* h_pfHTTurnOn_num_;
  MonitorElement* h_pfHTTurnOn_den_;

};

#endif

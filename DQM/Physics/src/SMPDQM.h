#ifndef SMPDQM_H
#define SMPDQM_H
#include <memory>

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <TH1.h>
#include <TH2.h>
#include "TFile.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonEnergy.h"
#include "DataFormats/MuonReco/interface/MuonIsolation.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "JetMETCorrections/JetCorrector/interface/JetCorrector.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "TLorentzVector.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include <DataFormats/METReco/interface/PFMET.h>

//
// class declaration
//

// If the analyzer does not use TFileService, please remove
// the template argument to the base class so the class inherits
// from  edm::one::EDAnalyzer<>
// This will improve performance in multithreaded jobs.

//using reco::TrackCollection;
class DQMStore;
class SMPDQM : public DQMEDAnalyzer {
public:
  SMPDQM(const edm::ParameterSet &);
  ~SMPDQM() override;

protected:
  void analyze(const edm::Event &, const edm::EventSetup &) override;

private:
  void bookHistograms(DQMStore::IBooker &bei, edm::Run const &, edm::EventSetup const &) override;
  void bookHistos(DQMStore *bei);

  edm::EDGetTokenT<reco::MuonCollection> muons_;
  edm::EDGetTokenT<reco::GsfElectronCollection> elecs_;
  edm::EDGetTokenT<edm::View<reco::Vertex> > pvs_;
  edm::EDGetTokenT<edm::View<reco::PFJet> > jets_;
  std::vector<edm::EDGetTokenT<edm::View<reco::MET> > > mets_;
  //edm::EDGetTokenT<reco::PFMETCollection> thePfMETCollectionToken_;
  // ----------member data ---------------------------

  //NPV
  MonitorElement *NPV;
  //MET
  MonitorElement *MET;
  MonitorElement *METphi;
  //muons
  MonitorElement *pt_muons;
  MonitorElement *eta_muons;
  MonitorElement *phi_muons;
  MonitorElement *muIso_CombRelIso03;
  MonitorElement *Nmuons;
  MonitorElement *isGlobalmuon;
  MonitorElement *isTrackermuon;
  MonitorElement *isStandalonemuon;
  MonitorElement *isPFmuon;
  MonitorElement *muIso_TrackerBased03;
  //electrons
  MonitorElement *Nelecs;
  MonitorElement *HoverE_elecs;
  MonitorElement *pt_elecs;
  MonitorElement *eta_elecs;
  MonitorElement *phi_elecs;
  MonitorElement *elIso_cal;
  MonitorElement *elIso_trk;
  MonitorElement *elIso_CombRelIso;
  //jets
  MonitorElement *PFJetpt;
  MonitorElement *PFJeteta;
  MonitorElement *PFJetphi;
  MonitorElement *PFJetMulti;
  MonitorElement *PFJetRapidity;
  MonitorElement *mjj;
  MonitorElement *detajj;
  //lepMET

  MonitorElement *dphi_lepMET;
  MonitorElement *mass_lepMET;
  MonitorElement *pt_lepMET;
  MonitorElement *detall;
  MonitorElement *dphill;
  MonitorElement *mll;
  MonitorElement *etall;
  MonitorElement *ptll;
  //lepjet1
  MonitorElement *dphi_lepjet1;
  MonitorElement *dphi_lep1jet1;
  MonitorElement *dphi_lep2jet1;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//

#endif
//
// member functions
//

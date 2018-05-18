#ifndef FSQDQM_H
#define FSQDQM_H

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

// Trigger stuff
#include "DataFormats/Common/interface/TriggerResults.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/DataKeyTags.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include <DataFormats/EgammaCandidates/interface/GsfElectron.h>

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "DQMServices/Core/interface/MonitorElement.h"

// ParticleFlow
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

// EGamma
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"

// Muon
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/MuonReco/interface/MuonIsolation.h"

// Jets
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/BasicJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"

// MET
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/METReco/interface/GenMETCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METCollection.h"

//
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "DataFormats/Math/interface/deltaPhi.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <cmath>

class DQMStore;

class FSQDQM : public DQMEDAnalyzer{

 public:
 FSQDQM(const edm::ParameterSet& ps);
 ~FSQDQM() override;

 protected:
  void analyze(edm::Event const& e, 
               edm::EventSetup const& eSetup) override;
 private:
  void bookHistograms(DQMStore::IBooker & bei, edm::Run const &, edm::EventSetup const &) override;
  void bookHistos(DQMStore* bei);



  edm::InputTag vertex_;
  std::string labelBS_, labelTrack_,labelPFJet_,labelCastorJet_;
  edm::EDGetTokenT<edm::View<reco::Vertex> >                pvs_;
  edm::EDGetTokenT<reco::TrackCollection>                tok_track_;
  edm::EDGetTokenT<reco::PFJetCollection> tok_pfjet_;
  edm::EDGetTokenT<reco::BasicJetCollection> tok_castorjet_;

  std::vector<int>          hltresults;
  unsigned int runNumber_, eventNumber_ , lumiNumber_, bxNumber_;

  //Histograms
  MonitorElement *PFJetpt;
  MonitorElement *PFJeteta;
  MonitorElement *PFJetphi;


  MonitorElement *CastorJetphi;
  MonitorElement *CastorJetMulti;
  MonitorElement *PFJetMulti;
  MonitorElement *PFJetRapidity;
  MonitorElement *Track_HP_Phi;
  MonitorElement *Track_HP_Eta;
  MonitorElement *Track_HP_Pt;
  MonitorElement *Track_HP_ptErr_over_pt;
  MonitorElement *Track_HP_dzvtx_over_dzerr;
  MonitorElement *Track_HP_dxyvtx_over_dxyerror;
  MonitorElement *NPV;
  MonitorElement *PV_chi2;
  MonitorElement *PV_d0;
  MonitorElement *PV_numTrks;
  MonitorElement *PV_sumTrks;
  MonitorElement *h_ptsum_towards;
  MonitorElement *h_ptsum_transverse;
  MonitorElement *h_ptsum_away;
  MonitorElement *h_ntracks_towards;
  MonitorElement *h_ntracks_transverse;
  MonitorElement *h_ntracks_away;
  MonitorElement *h_trkptsum;
  MonitorElement *h_ntracks;

  MonitorElement *h_leadingtrkpt_ntrk_away;
  MonitorElement *h_leadingtrkpt_ntrk_towards;
  MonitorElement *h_leadingtrkpt_ntrk_transverse;
  MonitorElement *h_leadingtrkpt_ptsum_away;
  MonitorElement *h_leadingtrkpt_ptsum_towards;
  MonitorElement *h_leadingtrkpt_ptsum_transverse;

  //  math::XYZPoint RefVtx;
};
#endif

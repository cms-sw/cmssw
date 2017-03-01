#include "DQM/Physics/src/HiggsDQM.h"

#include <memory>

// Framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Run.h"
#include "DataFormats/Provenance/interface/EventID.h"

// Candidate handling
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/OverlapChecker.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "DataFormats/Candidate/interface/CompositeCandidateFwd.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

//#include
//"HiggsAnalysis/HiggsToZZ4Leptons/plugins/HZZ4LeptonsElectronAssociationMap.h"
//#include
//"HiggsAnalysis/HiggsToZZ4Leptons/plugins/HZZ4LeptonsMuonAssociationMap.h"

// vertexing
// Transient tracks
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

// Vertex utilities
#include "DataFormats/VertexReco/interface/Vertex.h"

// Geometry
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Common/interface/RefToBase.h"

// RECO
// Isolation
//#include "HiggsAnalysis/HiggsToZZ4Leptons/plugins/CandidateHadIsolation.h"
//#include "HiggsAnalysis/HiggsToZZ4Leptons/plugins/CandidateTkIsolation.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonIsolation.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

// MET
#include "DataFormats/METReco/interface/GenMETCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/MET.h"

#include "DataFormats/Math/interface/LorentzVector.h"
#include "TLorentzVector.h"

#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <string>
#include <sstream>
#include <math.h>

using namespace edm;
using namespace std;
using namespace reco;

struct SortCandByDecreasingPt {
  bool operator()(const Candidate& c1, const Candidate& c2) const {
    return c1.pt() > c2.pt();
  }
};

double HiggsDQM::Distance(const reco::Candidate& c1,
                          const reco::Candidate& c2) {
  return deltaR(c1, c2);
}

double HiggsDQM::DistancePhi(const reco::Candidate& c1,
                             const reco::Candidate& c2) {
  return deltaPhi(c1.p4().phi(), c2.p4().phi());
}

// This always returns only a positive deltaPhi
double HiggsDQM::calcDeltaPhi(double phi1, double phi2) {
  double deltaPhi = phi1 - phi2;
  if (deltaPhi < 0) deltaPhi = -deltaPhi;
  if (deltaPhi > 3.1415926) {
    deltaPhi = 2 * 3.1415926 - deltaPhi;
  }
  return deltaPhi;
}

//
// -- Constructor
//
HiggsDQM::HiggsDQM(const edm::ParameterSet& ps) {
  // cout<<"Entering  HiggsDQM::HiggsDQM: "<<endl;

  edm::LogInfo("HZZ4LeptonsDQM") << " Creating HZZ4LeptonsDQM "
                                 << "\n";

  typedef std::vector<edm::InputTag> vtag;
  // Get parameters from configuration file
  theElecTriggerPathToPass = ps.getParameter<string>("elecTriggerPathToPass");
  theMuonTriggerPathToPass = ps.getParameter<string>("muonTriggerPathToPass");
  theTriggerResultsCollectionTag_ =
      ps.getParameter<InputTag>("triggerResultsCollection");
  theCaloJetCollectionLabel_ = ps.getParameter<InputTag>("caloJetCollection");
  theTriggerResultsCollection_ =
      consumes<edm::TriggerResults>(theTriggerResultsCollectionTag_);
  theMuonCollectionToken_ = consumes<reco::MuonCollection>(
      ps.getParameter<InputTag>("muonCollection"));
  theElectronCollectionToken_ = consumes<reco::GsfElectronCollection>(
      ps.getParameter<InputTag>("electronCollection"));
  theCaloJetCollectionToken_ =
      consumes<reco::CaloJetCollection>(theCaloJetCollectionLabel_);
  theCaloMETCollectionToken_ = consumes<reco::CaloMETCollection>(
      ps.getParameter<InputTag>("caloMETCollection"));
  thePfMETCollectionToken_ = consumes<reco::PFMETCollection>(
      ps.getParameter<InputTag>("pfMETCollection"));
  vertexToken_ =
      consumes<reco::VertexCollection>(ps.getUntrackedParameter<InputTag>(
          "vertexCollection", InputTag("offlinePrimaryVertices")));

  // cuts:
  ptThrMu1_ = ps.getUntrackedParameter<double>("PtThrMu1");
  ptThrMu2_ = ps.getUntrackedParameter<double>("PtThrMu2");

  nLumiSecs_ = 0;
  nEvents_ = 0;
  pi = 3.14159265;
  // cout<<"...leaving  HiggsDQM::HiggsDQM. "<<endl;
}
//
// -- Destructor
//
HiggsDQM::~HiggsDQM() {
  // cout<<"Entering HiggsDQM::~HiggsDQM: "<<endl;

  edm::LogInfo("HiggsDQM") << " Deleting HiggsDQM "
                           << "\n";

  // cout<<"...leaving HiggsDQM::~HiggsDQM. "<<endl;
}

//
// -- Begin  Luminosity Block
//
void HiggsDQM::beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg,
                                    edm::EventSetup const& context) {
  // cout<<"Entering HiggsDQM::beginLuminosityBlock: "<<endl;

  edm::LogInfo("HiggsDQM") << "[HiggsDQM]: Begin of LS transition";

  // cout<<"...leaving HiggsDQM::beginLuminosityBlock. "<<endl;
}
//
//  -- Book histograms
//
void HiggsDQM::bookHistograms(DQMStore::IBooker & ibooker,
  edm::Run const &, edm::EventSetup const & ){
  ibooker.setCurrentFolder("Physics/Higgs");

  h_vertex_number = ibooker.book1D("h_vertex_number",
      "Number of event vertices in collection", 10, -0.5, 9.5);
  h_vertex_chi2 = ibooker.book1D("h_vertex_chi2",
      "Event Vertex #chi^{2}/n.d.o.f.", 100, 0.0, 2.0);
  h_vertex_numTrks = ibooker.book1D("h_vertex_numTrks",
      "Event Vertex, number of tracks", 100, -0.5, 99.5);
  h_vertex_sumTrks = ibooker.book1D("h_vertex_sumTrks",
      "Event Vertex, sum of track pt", 100, 0.0, 100.0);
  h_vertex_d0 = ibooker.book1D("h_vertex_d0", "Event Vertex d0", 100, -10.0, 10.0);
  h_jet_et = ibooker.book1D("h_jet_et",
      "Jet with highest E_{T} (from " + theCaloJetCollectionLabel_.label() +
      ");E_{T}(1^{st} jet) (GeV)", 20, 0., 200.0);
  h_jet2_et = ibooker.book1D("h_jet2_et",
      "Jet with 2^{nd} highest E_{T} (from " + theCaloJetCollectionLabel_.label() +
      ");E_{T}(2^{nd} jet) (GeV)", 20, 0., 200.0);
  h_jet_count = ibooker.book1D("h_jet_count",
      "Number of " + theCaloJetCollectionLabel_.label() +
      " (E_{T} > 15 GeV);Number of Jets", 8, -0.5, 7.5);
  h_caloMet = ibooker.book1D("h_caloMet", "Calo Missing E_{T}; GeV", 20, 0.0, 100);
  h_caloMet_phi = ibooker.book1D("h_caloMet_phi",
      "Calo Missing E_{T} #phi;#phi(MET)", 35, -3.5, 3.5);
  h_pfMet = ibooker.book1D("h_pfMet", "Pf Missing E_{T}; GeV", 20, 0.0, 100);
  h_pfMet_phi = ibooker.book1D("h_pfMet_phi", "Pf Missing E_{T} #phi;#phi(MET)",
      35, -3.5, 3.5);
  h_eMultiplicity = ibooker.book1D("NElectrons",
      "# of electrons per event", 10, 0., 10.);
  h_mMultiplicity = ibooker.book1D("NMuons", "# of muons per event", 10, 0., 10.);
  h_ePt = ibooker.book1D("ElePt", "Pt of electrons", 50, 0., 100.);
  h_eEta = ibooker.book1D("EleEta", "Eta of electrons", 100, -5., 5.);
  h_ePhi = ibooker.book1D("ElePhi", "Phi of electrons", 100, -3.5, 3.5);
  h_mPt_GMTM = ibooker.book1D("MuonPt_GMTM",
      "Pt of global+tracker muons", 50, 0., 100.);
  h_mEta_GMTM = ibooker.book1D("MuonEta_GMTM",
      "Eta of global+tracker muons", 60, -3., 3.);
  h_mPhi_GMTM = ibooker.book1D("MuonPhi_GMTM",
      "Phi of global+tracker muons", 70, -3.5, 3.5);
  h_mPt_GMPT = ibooker.book1D("MuonPt_GMPT",
      "Pt of global prompt-tight muons", 50, 0., 100.);
  h_mEta_GMPT = ibooker.book1D("MuonEta_GMPT",
      "Eta of global prompt-tight muons", 60, -3., 3.);
  h_mPhi_GMPT = ibooker.book1D("MuonPhi_GMPT",
      "Phi of global prompt-tight muons", 70, -3.5, 3.5);
  h_mPt_GM = ibooker.book1D("MuonPt_GM", "Pt of global muons", 50, 0., 100.);
  h_mEta_GM = ibooker.book1D("MuonEta_GM", "Eta of global muons", 60, -3., 3.);
  h_mPhi_GM = ibooker.book1D("MuonPhi_GM", "Phi of global muons", 70, -3.5, 3.5);
  h_mPt_TM = ibooker.book1D("MuonPt_TM", "Pt of tracker muons", 50, 0., 100.);
  h_mEta_TM = ibooker.book1D("MuonEta_TM", "Eta of tracker muons", 60, -3., 3.);
  h_mPhi_TM = ibooker.book1D("MuonPhi_TM", "Phi of tracker muons", 70, -3.5, 3.5);
  h_mPt_STAM = ibooker.book1D("MuonPt_STAM", "Pt of STA muons", 50, 0., 100.);
  h_mEta_STAM = ibooker.book1D("MuonEta_STAM", "Eta of STA muons", 60, -3., 3.);
  h_mPhi_STAM = ibooker.book1D("MuonPhi_STAM", "Phi of STA muons", 70, -3.5, 3.5);
  h_eCombIso = ibooker.book1D("EleCombIso", "CombIso of electrons", 100, 0., 10.);
  h_mCombIso = ibooker.book1D("MuonCombIso", "CombIso of muons", 100, 0., 10.);
  h_dimumass_GMGM = ibooker.book1D("DimuMass_GMGM", "Invariant mass of GMGM pairs",
      100, 0., 200.);
  h_dimumass_GMTM = ibooker.book1D("DimuMass_GMTM",
      "Invariant mass of GMTM pairs", 100, 0., 200.);
  h_dimumass_TMTM = ibooker.book1D("DimuMass_TMTM",
      "Invariant mass of TMTM pairs", 100, 0., 200.);
  h_dielemass = ibooker.book1D("DieleMass",
      "Invariant mass of EE pairs", 100, 0., 200.);
  h_lepcounts = ibooker.book1D("LeptonCounts",
      "LeptonCounts for multi lepton events", 49, 0., 49.);

  ibooker.cd();
}
//
//  -- Analyze
//
void HiggsDQM::analyze(const edm::Event& e, const edm::EventSetup& eSetup) {
  // cout<<"[HiggsDQM::analyze()] "<<endl;

  //-------------------------------
  //--- Trigger Info
  //-------------------------------
  // Did it pass certain HLT path?
  bool passed_electron_HLT = true;
  bool passed_muon_HLT = true;

  //-------------------------------
  //--- Vertex Info
  //-------------------------------
  Handle<VertexCollection> vertexHandle;
  e.getByToken(vertexToken_, vertexHandle);
  if (vertexHandle.isValid()) {
    VertexCollection vertexCollection = *(vertexHandle.product());
    int vertex_number = vertexCollection.size();
    VertexCollection::const_iterator v = vertexCollection.begin();
    double vertex_chi2 = v->normalizedChi2();  // v->chi2();
    double vertex_d0 = sqrt(v->x() * v->x() + v->y() * v->y());
    // double vertex_ndof    = v->ndof();cout << "ndof="<<vertex_ndof<<endl;
    double vertex_numTrks = v->tracksSize();
    double vertex_sumTrks = 0.0;
    for (Vertex::trackRef_iterator vertex_curTrack = v->tracks_begin();
         vertex_curTrack != v->tracks_end(); vertex_curTrack++) {
      vertex_sumTrks += (*vertex_curTrack)->pt();
    }
    h_vertex_number->Fill(vertex_number);
    h_vertex_chi2->Fill(vertex_chi2);
    h_vertex_d0->Fill(vertex_d0);
    h_vertex_numTrks->Fill(vertex_numTrks);
    h_vertex_sumTrks->Fill(vertex_sumTrks);
  }

  //-------------------------------
  //--- Electrons
  //-------------------------------
  float nEle = 0;
  Handle<GsfElectronCollection> electronCollection;
  e.getByToken(theElectronCollectionToken_, electronCollection);
  if (electronCollection.isValid()) {
    int posEle = 0, negEle = 0;
    // If it passed electron HLT and the collection was found, find electrons
    // near Z mass
    if (passed_electron_HLT) {
      for (reco::GsfElectronCollection::const_iterator recoElectron =
               electronCollection->begin();
           recoElectron != electronCollection->end(); recoElectron++) {
        //      cout << "Electron with pt= " <<  recoElectron->pt() << " and
        // eta" << recoElectron->eta() << " p=" <<  recoElectron->p() << endl;
        h_ePt->Fill(recoElectron->pt());
        h_eEta->Fill(recoElectron->eta());
        h_ePhi->Fill(recoElectron->phi());
        if (recoElectron->charge() == 1) {
          posEle++;
        } else if (recoElectron->charge() == -1) {
          negEle++;
        }
        // Require electron to pass some basic cuts
        // if ( recoElectron->et() < 20 || fabs(recoElectron->eta())>2.5 )
        // continue;
        // Tighter electron cuts
        // if ( recoElectron->deltaPhiSuperClusterTrackAtVtx() > 0.58 ||
        //     recoElectron->deltaEtaSuperClusterTrackAtVtx() > 0.01 ||
        //     recoElectron->sigmaIetaIeta() > 0.027 ) continue;
      }  // end of loop over electrons
    }    // end if passed HLT
    nEle = posEle + negEle;
    if (nEle > 9.) nEle = 9.;
    h_eMultiplicity->Fill(nEle);

    // Z->ee:
    unsigned int eleCollectionSize = electronCollection->size();
    for (unsigned int i = 0; i < eleCollectionSize; i++) {
      const GsfElectron& ele = electronCollection->at(i);
      double pt = ele.pt();
      if (pt > ptThrMu1_) {
        for (unsigned int j = i + 1; j < eleCollectionSize; j++) {
          const GsfElectron& ele2 = electronCollection->at(j);
          double pt2 = ele2.pt();
          if (pt2 > ptThrMu2_) {
            const math::XYZTLorentzVector ZRecoEE(
                ele.px() + ele2.px(), ele.py() + ele2.py(),
                ele.pz() + ele2.pz(), ele.p() + ele2.p());
            h_dielemass->Fill(ZRecoEE.mass());
          }
        }
      }
    }
  }

  //-------------------------------
  //--- Muons
  //-------------------------------
  float nMu = 0;
  Handle<MuonCollection> muonCollection;
  e.getByToken(theMuonCollectionToken_, muonCollection);
  if (muonCollection.isValid()) {
    // Find the highest pt muons
    int posMu = 0, negMu = 0;
    TLorentzVector m1, m2;
    if (passed_muon_HLT) {
      for (reco::MuonCollection::const_iterator recoMuon =
               muonCollection->begin();
           recoMuon != muonCollection->end(); recoMuon++) {
        // cout << "Muon with pt= " <<  muIter->pt() << " and eta" <<
        // muIter->eta() << " p=" <<  muIter->p() << endl;
        if (recoMuon->isGlobalMuon() && recoMuon->isTrackerMuon()) {
          h_mPt_GMTM->Fill(recoMuon->pt());
          h_mEta_GMTM->Fill(recoMuon->eta());
          h_mPhi_GMTM->Fill(recoMuon->phi());
        } else if (recoMuon->isGlobalMuon() &&
                   (muon::isGoodMuon((*recoMuon),
                                     muon::GlobalMuonPromptTight))) {
          h_mPt_GMPT->Fill(recoMuon->pt());
          h_mEta_GMPT->Fill(recoMuon->eta());
          h_mPhi_GMPT->Fill(recoMuon->phi());
        } else if (recoMuon->isGlobalMuon()) {
          h_mPt_GM->Fill(recoMuon->pt());
          h_mEta_GM->Fill(recoMuon->eta());
          h_mPhi_GM->Fill(recoMuon->phi());
        } else if (recoMuon->isTrackerMuon() &&
                   (muon::segmentCompatibility(
                       (*recoMuon), reco::Muon::SegmentAndTrackArbitration))) {
          h_mPt_TM->Fill(recoMuon->pt());
          h_mEta_TM->Fill(recoMuon->eta());
          h_mPhi_TM->Fill(recoMuon->phi());
        } else if (recoMuon->isStandAloneMuon()) {
          h_mPt_STAM->Fill(recoMuon->pt());
          h_mEta_STAM->Fill(recoMuon->eta());
          h_mPhi_STAM->Fill(recoMuon->phi());
        }
        if (recoMuon->charge() == 1) {
          posMu++;
        } else if (recoMuon->charge() == -1) {
          negMu++;
        }
      }
      nMu = posMu + negMu;
      if (nMu > 9.) nMu = 9.;
      h_mMultiplicity->Fill(nMu);
    }

    // Z->mumu:
    unsigned int muonCollectionSize = muonCollection->size();
    for (unsigned int i = 0; i < muonCollectionSize; i++) {
      const Muon& mu = muonCollection->at(i);
      // if (!mu.isGlobalMuon()) continue;
      double pt = mu.pt();
      if (pt > ptThrMu1_) {
        for (unsigned int j = i + 1; j < muonCollectionSize; j++) {
          const Muon& mu2 = muonCollection->at(j);
          double pt2 = mu2.pt();
          if (pt2 > ptThrMu2_) {
            // Glb + Glb
            if (mu.isGlobalMuon() && mu2.isGlobalMuon()) {
              const math::XYZTLorentzVector ZRecoGMGM(
                  mu.px() + mu2.px(), mu.py() + mu2.py(), mu.pz() + mu2.pz(),
                  mu.p() + mu2.p());
              h_dimumass_GMGM->Fill(ZRecoGMGM.mass());
            }
            // Glb + TM
            else if (mu.isGlobalMuon() && mu2.isTrackerMuon()) {
              const math::XYZTLorentzVector ZRecoGMTM(
                  mu.px() + mu2.px(), mu.py() + mu2.py(), mu.pz() + mu2.pz(),
                  mu.p() + mu2.p());
              h_dimumass_GMTM->Fill(ZRecoGMTM.mass());
            }
            // TM + TM
            else if (mu.isTrackerMuon() && mu2.isTrackerMuon()) {
              const math::XYZTLorentzVector ZRecoTMTM(
                  mu.px() + mu2.px(), mu.py() + mu2.py(), mu.pz() + mu2.pz(),
                  mu.p() + mu2.p());
              h_dimumass_TMTM->Fill(ZRecoTMTM.mass());
            }
          }
        }
      }
    }
  }

  //-------------------------------
  //--- Jets
  //-------------------------------
  Handle<CaloJetCollection> caloJetCollection;
  e.getByToken(theCaloJetCollectionToken_, caloJetCollection);
  if (caloJetCollection.isValid()) {
    float jet_et = -8.0;
    //    float jet_eta   = -8.0; // UNUSED
    //    float jet_phi   = -8.0; // UNUSED
    int jet_count = 0;
    float jet2_et = -9.0;
    //    float jet2_eta  = -9.0; // UNUSED
    //    float jet2_phi  = -9.0; // UNUSED
    for (CaloJetCollection::const_iterator i_calojet =
             caloJetCollection->begin();
         i_calojet != caloJetCollection->end(); i_calojet++) {
      float jet_current_et = i_calojet->et();
      // if it overlaps with electron, it is not a jet
      // if ( electron_et>0.0 && fabs(i_calojet->eta()-electron_eta ) < 0.2 &&
      // calcDeltaPhi(i_calojet->phi(), electron_phi ) < 0.2) continue;
      // if ( electron2_et>0.0&& fabs(i_calojet->eta()-electron2_eta) < 0.2 &&
      // calcDeltaPhi(i_calojet->phi(), electron2_phi) < 0.2) continue;
      // if it has too low Et, throw away
      if (jet_current_et < 15) continue;
      jet_count++;
      if (jet_current_et > jet_et) {
        jet2_et = jet_et;  // 2nd highest jet get's et from current highest
                           //        jet2_eta = jet_eta; // UNUSED
                           //        jet2_phi = jet_phi; // UNUSED
        jet_et = i_calojet->et();  // current highest jet gets et from the new
                                   // highest
        //        jet_eta  = i_calojet->eta(); // UNUSED
        //        jet_phi  = i_calojet->phi(); // UNUSED
      } else if (jet_current_et > jet2_et) {
        jet2_et = i_calojet->et();
        //        jet2_eta = i_calojet->eta(); // UNUSED
        //        jet2_phi = i_calojet->phi(); // UNUSED
      }
    }
    if (jet_et > 0.0) {
      h_jet_et->Fill(jet_et);
      h_jet_count->Fill(jet_count);
    }
  }

  //-------------------------------
  //--- MET
  //-------------------------------
  Handle<CaloMETCollection> caloMETCollection;
  e.getByToken(theCaloMETCollectionToken_, caloMETCollection);
  if (caloMETCollection.isValid()) {
    float caloMet = caloMETCollection->begin()->et();
    float caloMet_phi = caloMETCollection->begin()->phi();
    h_caloMet->Fill(caloMet);
    h_caloMet_phi->Fill(caloMet_phi);
  }
  Handle<PFMETCollection> pfMETCollection;
  e.getByToken(thePfMETCollectionToken_, pfMETCollection);
  if (pfMETCollection.isValid()) {
    float pfMet = pfMETCollection->begin()->et();
    float pfMet_phi = pfMETCollection->begin()->phi();
    h_pfMet->Fill(pfMet);
    h_pfMet_phi->Fill(pfMet_phi);
  }

  //-------------------------------------
  //--- Events with more than 2 leptons:
  //-------------------------------------
  if (nMu + nEle > 2 && nMu + nEle < 10) {
    if (nMu == 0 && nEle == 3) h_lepcounts->Fill(0);
    if (nMu == 0 && nEle == 4) h_lepcounts->Fill(1);
    if (nMu == 0 && nEle == 5) h_lepcounts->Fill(2);
    if (nMu == 0 && nEle == 6) h_lepcounts->Fill(3);
    if (nMu == 0 && nEle == 7) h_lepcounts->Fill(4);
    if (nMu == 0 && nEle == 8) h_lepcounts->Fill(5);
    if (nMu == 0 && nEle == 9) h_lepcounts->Fill(6);
    if (nMu == 1 && nEle == 2) h_lepcounts->Fill(7);
    if (nMu == 1 && nEle == 3) h_lepcounts->Fill(8);
    if (nMu == 1 && nEle == 4) h_lepcounts->Fill(9);
    if (nMu == 1 && nEle == 5) h_lepcounts->Fill(10);
    if (nMu == 1 && nEle == 6) h_lepcounts->Fill(11);
    if (nMu == 1 && nEle == 7) h_lepcounts->Fill(12);
    if (nMu == 1 && nEle == 8) h_lepcounts->Fill(13);
    if (nMu == 2 && nEle == 1) h_lepcounts->Fill(14);
    if (nMu == 2 && nEle == 2) h_lepcounts->Fill(15);
    if (nMu == 2 && nEle == 3) h_lepcounts->Fill(16);
    if (nMu == 2 && nEle == 4) h_lepcounts->Fill(17);
    if (nMu == 2 && nEle == 5) h_lepcounts->Fill(18);
    if (nMu == 2 && nEle == 6) h_lepcounts->Fill(19);
    if (nMu == 2 && nEle == 7) h_lepcounts->Fill(20);
    if (nMu == 3 && nEle == 0) h_lepcounts->Fill(21);
    if (nMu == 3 && nEle == 1) h_lepcounts->Fill(22);
    if (nMu == 3 && nEle == 2) h_lepcounts->Fill(23);
    if (nMu == 3 && nEle == 3) h_lepcounts->Fill(24);
    if (nMu == 3 && nEle == 4) h_lepcounts->Fill(25);
    if (nMu == 3 && nEle == 5) h_lepcounts->Fill(26);
    if (nMu == 3 && nEle == 6) h_lepcounts->Fill(27);
    if (nMu == 4 && nEle == 0) h_lepcounts->Fill(28);
    if (nMu == 4 && nEle == 1) h_lepcounts->Fill(29);
    if (nMu == 4 && nEle == 2) h_lepcounts->Fill(30);
    if (nMu == 4 && nEle == 3) h_lepcounts->Fill(31);
    if (nMu == 4 && nEle == 4) h_lepcounts->Fill(32);
    if (nMu == 4 && nEle == 5) h_lepcounts->Fill(33);
    if (nMu == 5 && nEle == 0) h_lepcounts->Fill(34);
    if (nMu == 5 && nEle == 1) h_lepcounts->Fill(35);
    if (nMu == 5 && nEle == 2) h_lepcounts->Fill(36);
    if (nMu == 5 && nEle == 3) h_lepcounts->Fill(37);
    if (nMu == 5 && nEle == 4) h_lepcounts->Fill(38);
    if (nMu == 6 && nEle == 0) h_lepcounts->Fill(39);
    if (nMu == 6 && nEle == 1) h_lepcounts->Fill(40);
    if (nMu == 6 && nEle == 2) h_lepcounts->Fill(41);
    if (nMu == 6 && nEle == 3) h_lepcounts->Fill(42);
    if (nMu == 7 && nEle == 0) h_lepcounts->Fill(43);
    if (nMu == 7 && nEle == 1) h_lepcounts->Fill(44);
    if (nMu == 7 && nEle == 2) h_lepcounts->Fill(45);
    if (nMu == 8 && nEle == 0) h_lepcounts->Fill(46);
    if (nMu == 8 && nEle == 1) h_lepcounts->Fill(47);
    if (nMu == 9 && nEle == 0) h_lepcounts->Fill(48);
  }
  if ((nMu + nEle) >= 10)
    LogDebug("HiggsDQM") << "WARNING: " << nMu + nEle
                         << " leptons in this event: run=" << e.id().run()
                         << ", event=" << e.id().event() << "\n";
}
//
// -- End Luminosity Block
//
void HiggsDQM::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg,
                                  edm::EventSetup const& eSetup) {
  //  cout<<"Entering HiggsDQM::endLuminosityBlock: "<<endl;

  edm::LogInfo("HiggsDQM") << "[HiggsDQM]: End of LS transition, performing "
                              "the DQM client operation";

  nLumiSecs_++;
  // cout << "nLumiSecs_: "<< nLumiSecs_ << endl;

  edm::LogInfo("HiggsDQM")
      << "====================================================== " << endl
      << " ===> Iteration # " << nLumiSecs_ << " " << lumiSeg.luminosityBlock()
      << endl
      << "====================================================== " << endl;

  //  cout<<"...leaving HiggsDQM::endLuminosityBlock. "<<endl;
}
//
// -- End Run
//
void HiggsDQM::endRun(edm::Run const& run, edm::EventSetup const& eSetup) {
  //  cout<<"Entering HiggsDQM::endRun: "<<endl;

  // edm::LogVerbatim ("HiggsDQM") <<"[HiggsDQM]: End of Run, saving  DQM output
  // ";
  // int iRun = run.run();

  //  cout<<"...leaving HiggsDQM::endRun. "<<endl;
}

// Local Variables:
// show-trailing-whitespace: t
// truncate-lines: t
// End:

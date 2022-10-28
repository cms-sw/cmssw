#include "DQM/Physics/src/FSQDQM.h"
#include <memory>

// DQM
#include "DQMServices/Core/interface/DQMStore.h"

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

// Vertex utilities
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

// Other
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Common/interface/RefToBase.h"

// Math
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"

// vertexing

// Transient tracks
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

// Substructure
#include "RecoJets/JetAlgorithms/interface/CATopJetHelper.h"
#include "DataFormats/BTauReco/interface/CATopJetTagInfo.h"

// ROOT
#include "TLorentzVector.h"

// STDLIB
#include <iostream>
#include <iomanip>
#include <cstdio>
#include <string>
#include <sstream>
#include <cmath>

using namespace edm;
using namespace std;
using namespace reco;
using namespace trigger;

struct SortByPt

{
  bool operator()(const TLorentzVector& a, const TLorentzVector& b) const { return a.Pt() > b.Pt(); }
};

FSQDQM::FSQDQM(const edm::ParameterSet& iConfig) {
  edm::LogInfo("FSQDQM") << " Creating FSQDQM "
                         << "\n";
  pvs_ = consumes<edm::View<reco::Vertex> >(iConfig.getParameter<edm::InputTag>("pvs"));
  labelPFJet_ = iConfig.getParameter<std::string>("LabelPFJet");
  labelCastorJet_ = iConfig.getParameter<std::string>("LabelCastorJet");
  labelTrack_ = iConfig.getParameter<std::string>("LabelTrack");
  tok_pfjet_ = consumes<reco::PFJetCollection>(labelPFJet_);
  tok_track_ = consumes<reco::TrackCollection>(labelTrack_);
  tok_castorjet_ = consumes<reco::BasicJetCollection>(labelCastorJet_);
}

FSQDQM::~FSQDQM() {
  edm::LogInfo("FSQDQM") << " Deleting FSQDQM "
                         << "\n";
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}
void FSQDQM::bookHistograms(DQMStore::IBooker& bei, edm::Run const&, edm::EventSetup const&) {
  bei.setCurrentFolder("Physics/FSQ");
  PFJetpt = bei.book1D("PFJetpt", ";p_{T}(PFJet)", 100, 0.0, 100);
  PFJeteta = bei.book1D("PFJeteta", ";#eta(PFJet)", 50, -2.5, 2.5);
  PFJetphi = bei.book1D("PFJetphi", ";#phi(PFJet)", 50, -3.14, 3.14);
  PFJetMulti = bei.book1D("PFJetMulti", ";No. of PFJets", 10, -0.5, 9.5);
  PFJetRapidity = bei.book1D("PFJetRapidity", ";PFJetRapidity", 50, -6.0, 6.0);
  CastorJetphi = bei.book1D("CastorJetphi", ";#phi(CastorJet)", 50, -3.14, 3.14);
  CastorJetMulti = bei.book1D("CastorJetMulti", ";No. of CastorJets", 10, -0.5, 9.5);
  Track_HP_Phi = bei.book1D("Track_HP_Phi", ";#phi(HPtrack)", 50, -3.14, 3.14);
  Track_HP_Eta = bei.book1D("Track_HP_Eta", ";#eta(HPtrack)", 50, -2.5, 2.5);
  Track_HP_Pt = bei.book1D("Track_HP_Pt", ";p_{T}(HPtrack)", 500, 0.0, 50);
  Track_HP_ptErr_over_pt = bei.book1D("Track_HP_ptErr_over_pt", ";{p_{T}Err}/p_{T}", 100, 0, 0.1);
  Track_HP_dzvtx_over_dzerr = bei.book1D("Track_HP_dzvtx_over_dzerr", ";dZerr/dZ", 100, -10, 10);
  Track_HP_dxyvtx_over_dxyerror = bei.book1D("Track_HP_dxyvtx_over_dxyerror", ";dxyErr/dxy", 100, -10, 10);
  NPV = bei.book1D("NPV", ";NPV", 10, -0.5, 9.5);
  PV_chi2 = bei.book1D("PV_chi2", ";PV_chi2", 100, 0.0, 2.0);
  PV_d0 = bei.book1D("PV_d0", ";PV_d0", 100, -10.0, 10.0);
  PV_numTrks = bei.book1D("PV_numTrks", ";PV_numTrks", 100, -0.5, 99.5);
  PV_sumTrks = bei.book1D("PV_sumTrks", ";PV_sumTrks", 100, 0, 100);
  h_ptsum_towards = bei.book1D("h_ptsum_towards", ";h_ptsum_towards", 100, 0, 100);
  h_ptsum_transverse = bei.book1D("h_ptsum_transverse", ";h_ptsum_transverse", 100, 0, 100);

  h_ntracks = bei.book1D("h_ntracks", ";h_ntracks", 50, -0.5, 49.5);
  h_trkptsum = bei.book1D("h_trkptsum", ";h_trkptsum", 100, 0, 100);
  h_ptsum_away = bei.book1D("h_ptsum_away", ";h_ptsum_away", 100, 0, 100);
  h_ntracks_towards = bei.book1D("h_ntracks_towards", ";h_ntracks_towards", 50, -0.5, 49.5);
  h_ntracks_transverse = bei.book1D("h_ntracks_transverse", ";h_ntracks_transverse", 50, -0.5, 49.5);
  h_ntracks_away = bei.book1D("h_ntracks_away", ";h_ntracks_away", 50, -0.5, 49.5);

  h_leadingtrkpt_ntrk_away =
      bei.bookProfile("h_leadingtrkpt_ntrk_away", "h_leadingtrkpt_ntrk_away", 50, 0.0, 50, 0, 30, " ");
  h_leadingtrkpt_ntrk_towards =
      bei.bookProfile("h_leadingtrkpt_ntrk_towards", "h_leadingtrkpt_ntrk_towards", 50, 0.0, 50, 0, 30, " ");
  h_leadingtrkpt_ntrk_transverse =
      bei.bookProfile("h_leadingtrkpt_ntrk_transverse", "h_leadingtrkpt_ntrk_transverse", 50, 0.0, 50, 0, 30, " ");
  h_leadingtrkpt_ptsum_away =
      bei.bookProfile("h_leadingtrkpt_ptsum_away", "h_leadingtrkpt_ptsum_away", 50, 0.0, 50, 0, 30, " ");
  h_leadingtrkpt_ptsum_towards =
      bei.bookProfile("h_leadingtrkpt_ptsum_towards", "h_leadingtrkpt_ptsum_towards", 50, 0.0, 50, 0, 30, " ");
  h_leadingtrkpt_ptsum_transverse =
      bei.bookProfile("h_leadingtrkpt_ptsum_transverse", "h_leadingtrkpt_ptsum_transverse", 50, 0.0, 50, 0, 30, " ");
}

// ------------ method called for each event  ------------
void FSQDQM::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace std;
  using namespace reco;

  runNumber_ = iEvent.id().run();
  eventNumber_ = iEvent.id().event();
  lumiNumber_ = iEvent.id().luminosityBlock();
  bxNumber_ = iEvent.bunchCrossing();

  edm::Handle<edm::View<reco::Vertex> > privtxs;
  //    if(!iEvent.getByToken(pvs_, privtxs)) return;
  iEvent.getByToken(pvs_, privtxs);
  double bestvz = -999.9, bestvx = -999.9, bestvy = -999.9;
  double bestvzError = -999.9, bestvxError = -999.9, bestvyError = -999.9;
  if (privtxs.isValid()) {
    const reco::Vertex& pvtx = privtxs->front();
    NPV->Fill(privtxs->size());
    if (privtxs->begin() != privtxs->end() && !(pvtx.isFake()) && pvtx.position().Rho() <= 2. &&
        fabs(pvtx.position().z()) <= 24) {
      bestvz = pvtx.z();
      bestvx = pvtx.x();
      bestvy = pvtx.y();
      bestvzError = pvtx.zError();
      bestvxError = pvtx.xError();
      bestvyError = pvtx.yError();
      PV_chi2->Fill(pvtx.normalizedChi2());
      PV_d0->Fill(sqrt(pvtx.x() * pvtx.x() + pvtx.y() * pvtx.y()));
      PV_numTrks->Fill(pvtx.tracksSize());
      double vertex_sumTrks = 0.0;
      for (reco::Vertex::trackRef_iterator iTrack = pvtx.tracks_begin(); iTrack != pvtx.tracks_end(); iTrack++) {
        vertex_sumTrks += (*iTrack)->pt();
      }
      PV_sumTrks->Fill(vertex_sumTrks);
    }
  }

  std::vector<Jet> recoPFJets;
  recoPFJets.clear();
  int nPFCHSJet = 0;
  edm::Handle<PFJetCollection> pfjetchscoll;
  //	 if(!iEvent.getByToken(tok_pfjet_, pfjetchscoll)) return;
  iEvent.getByToken(tok_pfjet_, pfjetchscoll);
  if (pfjetchscoll.isValid()) {
    const reco::PFJetCollection* pfchsjets = pfjetchscoll.product();
    reco::PFJetCollection::const_iterator pfjetchsclus = pfchsjets->begin();
    for (pfjetchsclus = pfchsjets->begin(); pfjetchsclus != pfchsjets->end(); ++pfjetchsclus) {
      PFJetpt->Fill(pfjetchsclus->pt());
      PFJeteta->Fill(pfjetchsclus->eta());
      PFJetphi->Fill(pfjetchsclus->phi());
      PFJetRapidity->Fill(pfjetchsclus->rapidity());
      nPFCHSJet++;
    }
    PFJetMulti->Fill(nPFCHSJet);
  }

  std::vector<Jet> recoCastorJets;
  recoCastorJets.clear();
  edm::Handle<BasicJetCollection> castorJets;
  //	 if(!iEvent.getByToken(tok_castorjet_, castorJets)) return;
  iEvent.getByToken(tok_castorjet_, castorJets);
  if (castorJets.isValid()) {
    for (unsigned ijet = 0; ijet < castorJets->size(); ijet++) {
      recoCastorJets.push_back((*castorJets)[ijet]);
    }
    for (unsigned ijet = 0; ijet < recoCastorJets.size(); ijet++) {
      //	   cout<<recoCastorJets[ijet].pt()<<endl;
      CastorJetphi->Fill(recoCastorJets[ijet].phi());

      CastorJetMulti->Fill(recoCastorJets.size());
    }
  }
  edm::Handle<reco::TrackCollection> itracks;
  //	 if(!iEvent.getByToken(tok_track_, itracks)) return;
  iEvent.getByToken(tok_track_, itracks);
  if (itracks.isValid()) {
    reco::TrackBase::TrackQuality hiPurity = reco::TrackBase::qualityByName("highPurity");
    std::vector<TLorentzVector> T_trackRec_P4;

    int ntracks = 0;
    int ntracks_towards = 0;
    int ntracks_transverse = 0;
    int ntracks_away = 0;
    float ptsum = 0;
    float dphi = 0;
    float ptsum_towards = 0;
    float ptsum_transverse = 0;
    float ptsum_away = 0;

    T_trackRec_P4.clear();
    for (reco::TrackCollection::const_iterator iT = itracks->begin(); iT != itracks->end(); ++iT) {
      if (iT->quality(hiPurity)) {
        math::XYZPoint bestvtx(bestvx, bestvy, bestvz);
        double dzvtx = iT->dz(bestvtx);
        double dxyvtx = iT->dxy(bestvtx);
        double dzerror = sqrt(iT->dzError() * iT->dzError() + bestvzError * bestvzError);
        double dxyerror = sqrt(iT->d0Error() * iT->d0Error() + bestvxError * bestvyError);
        if ((iT->ptError()) / iT->pt() < 0.05 && dzvtx < 3.0 && dxyvtx < 3.0) {
          TLorentzVector trk;
          trk.SetPtEtaPhiE(iT->pt(), iT->eta(), iT->phi(), iT->p());
          T_trackRec_P4.push_back(trk);
          Track_HP_Eta->Fill(iT->eta());
          Track_HP_Phi->Fill(iT->phi());
          Track_HP_Pt->Fill(iT->pt());
          Track_HP_ptErr_over_pt->Fill((iT->ptError()) / iT->pt());
          Track_HP_dzvtx_over_dzerr->Fill(dzvtx / dzerror);
          Track_HP_dxyvtx_over_dxyerror->Fill(dxyvtx / dxyerror);
        }
      }
    }

    float highest_pt_track = -999;
    int index = -999;
    for (unsigned int k = 0; k < T_trackRec_P4.size(); k++) {
      if (T_trackRec_P4.at(k).Pt() > highest_pt_track) {
        highest_pt_track = T_trackRec_P4.at(k).Pt();
        index = k;
      }
    }
    unsigned int finalid = abs(index);
    //	 if(T_trackRec_P4.at(index).Pt()!=0.){
    if (finalid < T_trackRec_P4.size()) {
      //	 std::sort(T_trackRec_P4.begin(), T_trackRec_P4.end(), SortByPt());
      for (unsigned int itrk = 0; itrk < T_trackRec_P4.size(); itrk++) {
        ++ntracks;
        ptsum = ptsum + T_trackRec_P4.at(itrk).Pt();
        dphi = deltaPhi(T_trackRec_P4.at(itrk).Phi(), T_trackRec_P4.at(index).Phi());
        if (fabs(dphi) < 1.05) {
          ++ntracks_towards;
          ptsum_towards = ptsum_towards + T_trackRec_P4.at(itrk).Pt();
        }
        if (fabs(dphi) > 1.05 && fabs(dphi) < 2.09) {
          ++ntracks_transverse;
          ptsum_transverse = ptsum_transverse + T_trackRec_P4.at(itrk).Pt();
        }
        if (fabs(dphi) > 2.09) {
          ++ntracks_away;
          ptsum_away = ptsum_away + T_trackRec_P4.at(itrk).Pt();
        }
      }

      h_ntracks->Fill(ntracks);
      h_trkptsum->Fill(ptsum);
      h_ptsum_towards->Fill(ptsum_towards);
      h_ptsum_transverse->Fill(ptsum_transverse);
      h_ptsum_away->Fill(ptsum_away);
      h_ntracks_towards->Fill(ntracks_towards);
      h_ntracks_transverse->Fill(ntracks_transverse);
      h_ntracks_away->Fill(ntracks_away);

      if (!T_trackRec_P4.empty()) {
        h_leadingtrkpt_ntrk_towards->Fill(T_trackRec_P4.at(index).Pt(), ntracks_towards / 8.37);
        h_leadingtrkpt_ntrk_transverse->Fill(T_trackRec_P4.at(index).Pt(), ntracks_transverse / 8.37);
        h_leadingtrkpt_ntrk_away->Fill(T_trackRec_P4.at(index).Pt(), ntracks_away / 8.37);
        h_leadingtrkpt_ptsum_towards->Fill(T_trackRec_P4.at(index).Pt(), ptsum_towards / 8.37);
        h_leadingtrkpt_ptsum_transverse->Fill(T_trackRec_P4.at(index).Pt(), ptsum_transverse / 8.37);
        h_leadingtrkpt_ptsum_away->Fill(T_trackRec_P4.at(index).Pt(), ptsum_away / 8.37);
      }
    }
  }
}  //analyze

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------

//define this as a plug-in
//DEFINE_FWK_MODULE(FSQDQM);

//  LocalWords:  TH1F ptsum fs ntracks

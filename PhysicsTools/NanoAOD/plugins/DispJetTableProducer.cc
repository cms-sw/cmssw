// -*- C++ -*-
//
// Package:    PhysicsTools/EXOnanoAOD
// Class:      DispJetTableProducer
//
/**\class DispJetTableProducer

 Description: Additional variables for displaced vertices involving at least one lepton and other tracks

*/
//
// Original Author:  Kirill Skovpen
//         Created:  Sat, 1 Mar 2025 08:37:01 GMT
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"

// nanoAOD include files
#include "DataFormats/NanoAOD/interface/FlatTable.h"

// object specific include files
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

class DispJetTableProducer : public edm::stream::EDProducer<> {
protected:
  edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> ttbToken_;
  edm::EDGetTokenT<std::vector<pat::Electron>> electronTag_;
  edm::EDGetTokenT<std::vector<pat::Muon>> muonTag_;
  edm::EDGetTokenT<reco::VertexCollection> vtxTag_;
  edm::EDGetTokenT<reco::VertexCollection> secVtxTag_;

public:
  DispJetTableProducer(edm::ParameterSet const& params)
      : ttbToken_(esConsumes(edm::ESInputTag("", "TransientTrackBuilder"))),
        electronTag_(consumes<std::vector<pat::Electron>>(params.getParameter<edm::InputTag>("electrons"))),
        muonTag_(consumes<std::vector<pat::Muon>>(params.getParameter<edm::InputTag>("muons"))),
        vtxTag_(consumes<reco::VertexCollection>(params.getParameter<edm::InputTag>("primaryVertex"))),
        secVtxTag_(consumes<reco::VertexCollection>(params.getParameter<edm::InputTag>("secondaryVertex"))) {
    produces<nanoaod::FlatTable>("DispJetElectron");
    produces<nanoaod::FlatTable>("DispJetElectronVtx");
    produces<nanoaod::FlatTable>("DispJetElectronTrk");
    produces<nanoaod::FlatTable>("DispJetMuon");
    produces<nanoaod::FlatTable>("DispJetMuonVtx");
    produces<nanoaod::FlatTable>("DispJetMuonTrk");
  }

  ~DispJetTableProducer() override {}

  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override {
    const TransientTrackBuilder* theB = &iSetup.getData(ttbToken_);

    const std::vector<pat::Electron>& electrons = iEvent.get(electronTag_);
    const std::vector<pat::Muon>& muons = iEvent.get(muonTag_);
    const reco::VertexCollection& primaryVertices = iEvent.get(vtxTag_);
    const reco::VertexCollection& secVertices = iEvent.get(secVtxTag_);
    const auto& pv = primaryVertices.at(0);
    GlobalPoint primaryVertex(pv.x(), pv.y(), pv.z());

    unsigned int nElectrons = electrons.size();
    unsigned int nMuons = muons.size();

    std::vector<int> el_idx;
    std::vector<bool> el_lIVF_match;
    std::vector<int> el_IVF_df, el_IVF_ntracks, el_IVF_elid;
    std::vector<float> el_IVF_x, el_IVF_y, el_IVF_z, el_IVF_cx, el_IVF_cy, el_IVF_cz, el_IVF_chi2, el_IVF_pt,
        el_IVF_eta, el_IVF_phi, el_IVF_E, el_IVF_mass;
    std::vector<int> el_IVF_trackcharge, el_IVF_trackelid, el_IVF_trackvtxid;
    std::vector<float> el_IVF_trackpt, el_IVF_tracketa, el_IVF_trackphi, el_IVF_trackE, el_IVF_trackdxy, el_IVF_trackdz;
    std::vector<float> el_IVF_tracksignedIP2D, el_IVF_tracksignedIP3D, el_IVF_tracksignedIP2Dsig,
        el_IVF_tracksignedIP3Dsig;
    std::vector<float> el_IVF_signedIP2D, el_IVF_signedIP3D, el_IVF_signedIP2Dsig, el_IVF_signedIP3Dsig;

    std::vector<int> mu_idx;
    std::vector<bool> mu_lIVF_match;
    std::vector<int> mu_IVF_df, mu_IVF_ntracks, mu_IVF_muid;
    std::vector<float> mu_IVF_x, mu_IVF_y, mu_IVF_z, mu_IVF_cx, mu_IVF_cy, mu_IVF_cz, mu_IVF_chi2, mu_IVF_pt,
        mu_IVF_eta, mu_IVF_phi, mu_IVF_E, mu_IVF_mass;
    std::vector<int> mu_IVF_trackcharge, mu_IVF_trackmuid, mu_IVF_trackvtxid;
    std::vector<float> mu_IVF_trackpt, mu_IVF_tracketa, mu_IVF_trackphi, mu_IVF_trackE, mu_IVF_trackdxy, mu_IVF_trackdz;
    std::vector<float> mu_IVF_tracksignedIP2D, mu_IVF_tracksignedIP3D, mu_IVF_tracksignedIP2Dsig,
        mu_IVF_tracksignedIP3Dsig;
    std::vector<float> mu_IVF_signedIP2D, mu_IVF_signedIP3D, mu_IVF_signedIP2Dsig, mu_IVF_signedIP3Dsig;

    el_idx.clear();
    el_lIVF_match.clear();
    el_IVF_df.clear();
    el_IVF_ntracks.clear();
    el_IVF_elid.clear();
    el_IVF_x.clear();
    el_IVF_y.clear();
    el_IVF_z.clear();
    el_IVF_cx.clear();
    el_IVF_cy.clear();
    el_IVF_cz.clear();
    el_IVF_chi2.clear();
    el_IVF_pt.clear();
    el_IVF_eta.clear();
    el_IVF_phi.clear();
    el_IVF_E.clear();
    el_IVF_mass.clear();
    el_IVF_trackcharge.clear();
    el_IVF_trackelid.clear();
    el_IVF_trackvtxid.clear();
    el_IVF_trackpt.clear();
    el_IVF_tracketa.clear();
    el_IVF_trackphi.clear();
    el_IVF_trackE.clear();
    el_IVF_trackdxy.clear();
    el_IVF_trackdz.clear();
    el_IVF_tracksignedIP2D.clear();
    el_IVF_tracksignedIP2Dsig.clear();
    el_IVF_tracksignedIP3D.clear();
    el_IVF_tracksignedIP3Dsig.clear();

    el_IVF_signedIP2D.clear();
    el_IVF_signedIP2Dsig.clear();
    el_IVF_signedIP3D.clear();
    el_IVF_signedIP3Dsig.clear();

    mu_idx.clear();
    mu_lIVF_match.clear();
    mu_IVF_df.clear();
    mu_IVF_ntracks.clear();
    mu_IVF_muid.clear();
    mu_IVF_x.clear();
    mu_IVF_y.clear();
    mu_IVF_z.clear();
    mu_IVF_cx.clear();
    mu_IVF_cy.clear();
    mu_IVF_cz.clear();
    mu_IVF_chi2.clear();
    mu_IVF_pt.clear();
    mu_IVF_eta.clear();
    mu_IVF_phi.clear();
    mu_IVF_E.clear();
    mu_IVF_mass.clear();
    mu_IVF_trackcharge.clear();
    mu_IVF_trackmuid.clear();
    mu_IVF_trackvtxid.clear();
    mu_IVF_trackpt.clear();
    mu_IVF_tracketa.clear();
    mu_IVF_trackphi.clear();
    mu_IVF_trackE.clear();
    mu_IVF_trackdxy.clear();
    mu_IVF_trackdz.clear();
    mu_IVF_tracksignedIP2D.clear();
    mu_IVF_tracksignedIP2Dsig.clear();
    mu_IVF_tracksignedIP3D.clear();
    mu_IVF_tracksignedIP3Dsig.clear();

    mu_IVF_signedIP2D.clear();
    mu_IVF_signedIP2Dsig.clear();
    mu_IVF_signedIP3D.clear();
    mu_IVF_signedIP3Dsig.clear();

    int ntrack_max = 100;
    int nElectronsSel = 0;
    int nMuonsSel = 0;

    for (unsigned int i = 0; i < nElectrons; i++) {
      const pat::Electron& el = electrons[i];

      if (el.gsfTrack().isNull())
        continue;
      if (el.pt() < 7)
        continue;
      if (fabs(el.eta()) > 2.5)
        continue;

      el_idx.push_back(i);
      el_lIVF_match.push_back(false);

      bool new_vtx = false;
      double dR, deta, normchi2;
      double mindR = 20, minnormchi2 = 10000;
      int nVtx = 0;
      reco::Vertex* vtxDisp = nullptr;
      for (const reco::Vertex& vtx : secVertices) {
        for (reco::Vertex::trackRef_iterator vtxTrackref = vtx.tracks_begin(); vtxTrackref != vtx.tracks_end();
             vtxTrackref++) {
          reco::TrackRef vtxTrack = vtxTrackref->castTo<reco::TrackRef>();
          for (edm::Ref<pat::PackedCandidateCollection> cand : el.associatedPackedPFCandidates()) {
            dR = reco::deltaR(cand->eta(), cand->phi(), vtxTrack->eta(), vtxTrack->phi());
            deta = fabs(cand->eta() - vtxTrack->eta());
            normchi2 = fabs(vtx.chi2() / vtx.ndof());

            if ((dR < 0.05 or (dR < 0.1 and deta < 0.03)) and
                (dR < mindR or (dR == mindR and normchi2 < minnormchi2))) {
              new_vtx = true;
              vtxDisp = const_cast<reco::Vertex*>(&vtx);
              el_lIVF_match[nElectronsSel] = true;
              mindR = dR;
              minnormchi2 = normchi2;
            }
          }
        }

        if (new_vtx) {
          el_IVF_x.push_back(vtx.x());
          el_IVF_y.push_back(vtx.y());
          el_IVF_z.push_back(vtx.z());
          el_IVF_cx.push_back(vtx.xError());
          el_IVF_cy.push_back(vtx.yError());
          el_IVF_cz.push_back(vtx.zError());
          el_IVF_df.push_back(vtx.ndof());
          el_IVF_chi2.push_back(vtx.chi2());
          el_IVF_pt.push_back(vtx.p4().pt());
          el_IVF_eta.push_back(vtx.p4().eta());
          el_IVF_phi.push_back(vtx.p4().phi());
          el_IVF_E.push_back(vtx.p4().energy());
          el_IVF_mass.push_back(vtx.p4().mass());
          el_IVF_elid.push_back(nElectronsSel);

          el_IVF_ntracks.push_back(0);
          for (reco::Vertex::trackRef_iterator vtxTrackref = vtx.tracks_begin(); vtxTrackref != vtx.tracks_end();
               vtxTrackref++) {
            if (el_IVF_ntracks.back() == ntrack_max)
              break;
            reco::TrackRef vtxTrack = vtxTrackref->castTo<reco::TrackRef>();
            const auto& trk = theB->build(vtxTrack);
            Global3DVector dir(vtxTrack->px(), vtxTrack->py(), vtxTrack->pz());
            const auto& ip2d = IPTools::signedTransverseImpactParameter(trk, dir, *vtxDisp);
            const auto& ip3d = IPTools::signedImpactParameter3D(trk, dir, *vtxDisp);
            el_IVF_tracksignedIP2D.push_back(ip2d.second.value());
            el_IVF_tracksignedIP3D.push_back(ip3d.second.value());
            el_IVF_tracksignedIP2Dsig.push_back(ip2d.second.significance());
            el_IVF_tracksignedIP3Dsig.push_back(ip3d.second.significance());
            el_IVF_trackpt.push_back(vtxTrack->pt());
            el_IVF_tracketa.push_back(vtxTrack->eta());
            el_IVF_trackphi.push_back(vtxTrack->phi());
            el_IVF_trackE.push_back(vtxTrack->p());
            el_IVF_trackcharge.push_back(vtxTrack->charge());
            el_IVF_trackdxy.push_back(std::abs(vtxTrack->dxy(pv.position())));
            el_IVF_trackdz.push_back(std::abs(vtxTrack->dz(pv.position())));
            el_IVF_trackelid.push_back(nElectronsSel);
            el_IVF_trackvtxid.push_back(nVtx);
            el_IVF_ntracks.back()++;
          }
          nVtx++;
          new_vtx = false;

          auto track = el.gsfTrack();
          if (track.isNonnull()) {
            const auto& trk = theB->build(track);
            Global3DVector dir(track->px(), track->py(), track->pz());
            const auto& ip2d = IPTools::signedTransverseImpactParameter(trk, dir, *vtxDisp);
            const auto& ip3d = IPTools::signedImpactParameter3D(trk, dir, *vtxDisp);
            el_IVF_signedIP2D.push_back(ip2d.second.value());
            el_IVF_signedIP3D.push_back(ip3d.second.value());
            el_IVF_signedIP2Dsig.push_back(ip2d.second.significance());
            el_IVF_signedIP3Dsig.push_back(ip3d.second.significance());
          } else {
            el_IVF_signedIP2D.push_back(-999);
            el_IVF_signedIP3D.push_back(-999);
            el_IVF_signedIP2Dsig.push_back(-999);
            el_IVF_signedIP3Dsig.push_back(-999);
          }
        }
      }
      nElectronsSel += 1;
    }

    for (unsigned int i = 0; i < nMuons; i++) {
      const pat::Muon& mu = muons[i];

      if (mu.innerTrack().isNull())
        continue;
      if (mu.pt() < 5)
        continue;
      if (fabs(mu.eta()) > 2.4)
        continue;
      if (!mu.isPFMuon())
        continue;
      if (!(mu.isTrackerMuon() || mu.isGlobalMuon()))
        continue;

      mu_idx.push_back(i);
      mu_lIVF_match.push_back(false);

      bool new_vtx = false;
      double ptdiff, normchi2;
      double minptdiff = 10, minnormchi2 = 10000;
      int nVtx = 0;
      reco::Vertex* vtxDisp = nullptr;
      for (const reco::Vertex& vtx : secVertices) {
        for (reco::Vertex::trackRef_iterator vtxTrackref = vtx.tracks_begin(); vtxTrackref != vtx.tracks_end();
             vtxTrackref++) {
          reco::TrackRef vtxTrack = vtxTrackref->castTo<reco::TrackRef>();
          for (size_t iCand = 0; iCand < mu.numberOfSourceCandidatePtrs(); ++iCand) {
            if (!(mu.sourceCandidatePtr(iCand).isNonnull() and mu.sourceCandidatePtr(iCand).isAvailable()))
              continue;
            ptdiff = fabs(mu.sourceCandidatePtr(iCand)->pt() - vtxTrack->pt());
            normchi2 = fabs(vtx.chi2() / vtx.ndof());

            if (ptdiff < 0.001 and (ptdiff < minptdiff or (ptdiff == minptdiff and normchi2 < minnormchi2))) {
              new_vtx = true;
              vtxDisp = const_cast<reco::Vertex*>(&vtx);
              mu_lIVF_match[nMuonsSel] = true;
              minptdiff = ptdiff;
              minnormchi2 = normchi2;
            }
          }
        }
        if (new_vtx) {
          mu_IVF_x.push_back(vtx.x());
          mu_IVF_y.push_back(vtx.y());
          mu_IVF_z.push_back(vtx.z());
          mu_IVF_cx.push_back(vtx.xError());
          mu_IVF_cy.push_back(vtx.yError());
          mu_IVF_cz.push_back(vtx.zError());
          mu_IVF_df.push_back(vtx.ndof());
          mu_IVF_chi2.push_back(vtx.chi2());
          mu_IVF_pt.push_back(vtx.p4().pt());
          mu_IVF_eta.push_back(vtx.p4().eta());
          mu_IVF_phi.push_back(vtx.p4().phi());
          mu_IVF_E.push_back(vtx.p4().energy());
          mu_IVF_mass.push_back(vtx.p4().mass());
          mu_IVF_muid.push_back(nMuonsSel);

          mu_IVF_ntracks.push_back(0);
          for (reco::Vertex::trackRef_iterator vtxTrackref = vtx.tracks_begin(); vtxTrackref != vtx.tracks_end();
               vtxTrackref++) {
            if (mu_IVF_ntracks.back() == ntrack_max)
              break;
            reco::TrackRef vtxTrack = vtxTrackref->castTo<reco::TrackRef>();
            const auto& trk = theB->build(vtxTrack);
            Global3DVector dir(vtxTrack->px(), vtxTrack->py(), vtxTrack->pz());
            const auto& ip2d = IPTools::signedTransverseImpactParameter(trk, dir, *vtxDisp);
            const auto& ip3d = IPTools::signedImpactParameter3D(trk, dir, *vtxDisp);
            mu_IVF_tracksignedIP2D.push_back(ip2d.second.value());
            mu_IVF_tracksignedIP3D.push_back(ip3d.second.value());
            mu_IVF_tracksignedIP2Dsig.push_back(ip2d.second.significance());
            mu_IVF_tracksignedIP3Dsig.push_back(ip3d.second.significance());
            mu_IVF_trackpt.push_back(vtxTrack->pt());
            mu_IVF_tracketa.push_back(vtxTrack->eta());
            mu_IVF_trackphi.push_back(vtxTrack->phi());
            mu_IVF_trackE.push_back(vtxTrack->p());
            mu_IVF_trackcharge.push_back(vtxTrack->charge());
            mu_IVF_trackdxy.push_back(std::abs(vtxTrack->dxy(pv.position())));
            mu_IVF_trackdz.push_back(std::abs(vtxTrack->dz(pv.position())));
            mu_IVF_trackmuid.push_back(nMuonsSel);
            mu_IVF_trackvtxid.push_back(nVtx);
            mu_IVF_ntracks.back()++;
          }
          nVtx++;
          new_vtx = false;

          auto track = mu.innerTrack();
          if (track.isNonnull()) {
            const auto& trk = theB->build(track);
            Global3DVector dir(track->px(), track->py(), track->pz());
            const auto& ip2d = IPTools::signedTransverseImpactParameter(trk, dir, *vtxDisp);
            const auto& ip3d = IPTools::signedImpactParameter3D(trk, dir, *vtxDisp);
            mu_IVF_signedIP2D.push_back(ip2d.second.value());
            mu_IVF_signedIP3D.push_back(ip3d.second.value());
            mu_IVF_signedIP2Dsig.push_back(ip2d.second.significance());
            mu_IVF_signedIP3Dsig.push_back(ip3d.second.significance());
          } else {
            mu_IVF_signedIP2D.push_back(-999);
            mu_IVF_signedIP3D.push_back(-999);
            mu_IVF_signedIP2Dsig.push_back(-999);
            mu_IVF_signedIP3Dsig.push_back(-999);
          }
        }
      }
      nMuonsSel += 1;
    }

    auto dispJetElectronTab = std::make_unique<nanoaod::FlatTable>(nElectronsSel, "DispJetElectron", false, false);
    auto dispJetMuonTab = std::make_unique<nanoaod::FlatTable>(nMuonsSel, "DispJetMuon", false, false);

    dispJetElectronTab->addColumn<int>("idx", el_idx, "");
    dispJetElectronTab->addColumn<bool>("lIVF_match", el_lIVF_match, "");

    auto dispJetElectronVtxTab =
        std::make_unique<nanoaod::FlatTable>(el_IVF_x.size(), "DispJetElectronVtx", false, false);
    dispJetElectronVtxTab->addColumn<int>("IVF_df", el_IVF_df, "");
    dispJetElectronVtxTab->addColumn<int>("IVF_ntracks", el_IVF_ntracks, "");
    dispJetElectronVtxTab->addColumn<int>("IVF_elid", el_IVF_elid, "");
    dispJetElectronVtxTab->addColumn<float>("IVF_x", el_IVF_x, "");
    dispJetElectronVtxTab->addColumn<float>("IVF_y", el_IVF_y, "");
    dispJetElectronVtxTab->addColumn<float>("IVF_z", el_IVF_z, "");
    dispJetElectronVtxTab->addColumn<float>("IVF_cx", el_IVF_cx, "");
    dispJetElectronVtxTab->addColumn<float>("IVF_cy", el_IVF_cy, "");
    dispJetElectronVtxTab->addColumn<float>("IVF_cz", el_IVF_cz, "");
    dispJetElectronVtxTab->addColumn<float>("IVF_chi2", el_IVF_chi2, "");
    dispJetElectronVtxTab->addColumn<float>("IVF_pt", el_IVF_pt, "");
    dispJetElectronVtxTab->addColumn<float>("IVF_eta", el_IVF_eta, "");
    dispJetElectronVtxTab->addColumn<float>("IVF_phi", el_IVF_phi, "");
    dispJetElectronVtxTab->addColumn<float>("IVF_E", el_IVF_E, "");
    dispJetElectronVtxTab->addColumn<float>("IVF_mass", el_IVF_mass, "");
    dispJetElectronVtxTab->addColumn<float>("IVF_signedIP2D", el_IVF_signedIP2D, "");
    dispJetElectronVtxTab->addColumn<float>("IVF_signedIP2Dsig", el_IVF_signedIP2Dsig, "");
    dispJetElectronVtxTab->addColumn<float>("IVF_signedIP3D", el_IVF_signedIP3D, "");
    dispJetElectronVtxTab->addColumn<float>("IVF_signedIP3Dsig", el_IVF_signedIP3Dsig, "");

    int nTracksElectron = 0;
    for (unsigned int iv = 0; iv < el_IVF_ntracks.size(); iv++) {
      nTracksElectron += std::min(el_IVF_ntracks[iv], ntrack_max);
    }
    auto dispJetElectronTrkTab =
        std::make_unique<nanoaod::FlatTable>(nTracksElectron, "DispJetElectronTrk", false, false);
    dispJetElectronTrkTab->addColumn<int>("IVF_trackcharge", el_IVF_trackcharge, "");
    dispJetElectronTrkTab->addColumn<float>("IVF_trackpt", el_IVF_trackpt, "");
    dispJetElectronTrkTab->addColumn<float>("IVF_tracketa", el_IVF_tracketa, "");
    dispJetElectronTrkTab->addColumn<float>("IVF_trackphi", el_IVF_trackphi, "");
    dispJetElectronTrkTab->addColumn<float>("IVF_trackE", el_IVF_trackE, "");
    dispJetElectronTrkTab->addColumn<float>("IVF_trackdxy", el_IVF_trackdxy, "");
    dispJetElectronTrkTab->addColumn<float>("IVF_trackdz", el_IVF_trackdz, "");
    dispJetElectronTrkTab->addColumn<int>("IVF_trackelid", el_IVF_trackelid, "");
    dispJetElectronTrkTab->addColumn<int>("IVF_trackvtxid", el_IVF_trackvtxid, "");
    dispJetElectronTrkTab->addColumn<float>("IVF_tracksignedIP2D", el_IVF_tracksignedIP2D, "");
    dispJetElectronTrkTab->addColumn<float>("IVF_tracksignedIP2Dsig", el_IVF_tracksignedIP2Dsig, "");
    dispJetElectronTrkTab->addColumn<float>("IVF_tracksignedIP3D", el_IVF_tracksignedIP3D, "");
    dispJetElectronTrkTab->addColumn<float>("IVF_tracksignedIP3Dsig", el_IVF_tracksignedIP3Dsig, "");

    dispJetMuonTab->addColumn<int>("idx", mu_idx, "");
    dispJetMuonTab->addColumn<bool>("lIVF_match", mu_lIVF_match, "");

    auto dispJetMuonVtxTab = std::make_unique<nanoaod::FlatTable>(mu_IVF_x.size(), "DispJetMuonVtx", false, false);
    dispJetMuonVtxTab->addColumn<int>("IVF_df", mu_IVF_df, "");
    dispJetMuonVtxTab->addColumn<int>("IVF_ntracks", mu_IVF_ntracks, "");
    dispJetMuonVtxTab->addColumn<int>("IVF_muid", mu_IVF_muid, "");
    dispJetMuonVtxTab->addColumn<float>("IVF_x", mu_IVF_x, "");
    dispJetMuonVtxTab->addColumn<float>("IVF_y", mu_IVF_y, "");
    dispJetMuonVtxTab->addColumn<float>("IVF_z", mu_IVF_z, "");
    dispJetMuonVtxTab->addColumn<float>("IVF_cx", mu_IVF_cx, "");
    dispJetMuonVtxTab->addColumn<float>("IVF_cy", mu_IVF_cy, "");
    dispJetMuonVtxTab->addColumn<float>("IVF_cz", mu_IVF_cz, "");
    dispJetMuonVtxTab->addColumn<float>("IVF_chi2", mu_IVF_chi2, "");
    dispJetMuonVtxTab->addColumn<float>("IVF_pt", mu_IVF_pt, "");
    dispJetMuonVtxTab->addColumn<float>("IVF_eta", mu_IVF_eta, "");
    dispJetMuonVtxTab->addColumn<float>("IVF_phi", mu_IVF_phi, "");
    dispJetMuonVtxTab->addColumn<float>("IVF_E", mu_IVF_E, "");
    dispJetMuonVtxTab->addColumn<float>("IVF_mass", mu_IVF_mass, "");
    dispJetMuonVtxTab->addColumn<float>("IVF_signedIP2D", mu_IVF_signedIP2D, "");
    dispJetMuonVtxTab->addColumn<float>("IVF_signedIP2Dsig", mu_IVF_signedIP2Dsig, "");
    dispJetMuonVtxTab->addColumn<float>("IVF_signedIP3D", mu_IVF_signedIP3D, "");
    dispJetMuonVtxTab->addColumn<float>("IVF_signedIP3Dsig", mu_IVF_signedIP3Dsig, "");

    int nTracksMuon = 0;
    for (unsigned int iv = 0; iv < mu_IVF_ntracks.size(); iv++) {
      nTracksMuon += std::min(mu_IVF_ntracks[iv], ntrack_max);
    }
    auto dispJetMuonTrkTab = std::make_unique<nanoaod::FlatTable>(nTracksMuon, "DispJetMuonTrk", false, false);
    dispJetMuonTrkTab->addColumn<int>("IVF_trackcharge", mu_IVF_trackcharge, "");
    dispJetMuonTrkTab->addColumn<float>("IVF_trackpt", mu_IVF_trackpt, "");
    dispJetMuonTrkTab->addColumn<float>("IVF_tracketa", mu_IVF_tracketa, "");
    dispJetMuonTrkTab->addColumn<float>("IVF_trackphi", mu_IVF_trackphi, "");
    dispJetMuonTrkTab->addColumn<float>("IVF_trackE", mu_IVF_trackE, "");
    dispJetMuonTrkTab->addColumn<float>("IVF_trackdxy", mu_IVF_trackdxy, "");
    dispJetMuonTrkTab->addColumn<float>("IVF_trackdz", mu_IVF_trackdz, "");
    dispJetMuonTrkTab->addColumn<int>("IVF_trackmuid", mu_IVF_trackmuid, "");
    dispJetMuonTrkTab->addColumn<int>("IVF_trackvtxid", mu_IVF_trackvtxid, "");
    dispJetMuonTrkTab->addColumn<float>("IVF_tracksignedIP2D", mu_IVF_tracksignedIP2D, "");
    dispJetMuonTrkTab->addColumn<float>("IVF_tracksignedIP2Dsig", mu_IVF_tracksignedIP2Dsig, "");
    dispJetMuonTrkTab->addColumn<float>("IVF_tracksignedIP3D", mu_IVF_tracksignedIP3D, "");
    dispJetMuonTrkTab->addColumn<float>("IVF_tracksignedIP3Dsig", mu_IVF_tracksignedIP3Dsig, "");

    iEvent.put(std::move(dispJetElectronTab), "DispJetElectron");
    iEvent.put(std::move(dispJetElectronVtxTab), "DispJetElectronVtx");
    iEvent.put(std::move(dispJetElectronTrkTab), "DispJetElectronTrk");
    iEvent.put(std::move(dispJetMuonTab), "DispJetMuon");
    iEvent.put(std::move(dispJetMuonVtxTab), "DispJetMuonVtx");
    iEvent.put(std::move(dispJetMuonTrkTab), "DispJetMuonTrk");
  }
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DispJetTableProducer);

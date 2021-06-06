// -*- C++ -*-
//
// Package:    EvtPlaneProducer
// Class:      EvtPlaneProducer
//
/**\class EvtPlaneProducer EvtPlaneProducer.cc RecoHI/EvtPlaneProducer/src/EvtPlaneProducer.cc

Description: <one line class summary>

Implementation:
<Notes on implementation>
*/
//
// Original Author:  Sergey Petrushanko
//         Created:  Fri Jul 11 10:05:00 2008
//
//

// system include files
#include <memory>
#include <iostream>
#include <ctime>
#include <cmath>
#include <cstdlib>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HeavyIonEvent/interface/EvtPlane.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/Common/interface/Ref.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/CastorReco/interface/CastorTower.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "RecoHI/HiEvtPlaneAlgos/interface/HiEvtPlaneList.h"
#include "CondFormats/HIObjects/interface/RPFlatParams.h"
#include "CondFormats/DataRecord/interface/HeavyIonRPRcd.h"
#include "CondFormats/DataRecord/interface/HeavyIonRcd.h"
#include "CondFormats/HIObjects/interface/CentralityTable.h"

#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "RecoHI/HiEvtPlaneAlgos/interface/HiEvtPlaneFlatten.h"
#include "RecoHI/HiEvtPlaneAlgos/interface/LoadEPDB.h"
#include "RecoHI/HiEvtPlaneAlgos/interface/EPCuts.h"

using namespace std;
using namespace hi;

//
// class decleration
//

namespace hi {
  class GenPlane {
  public:
    GenPlane(string name, double etaminval1, double etamaxval1, double etaminval2, double etamaxval2, int orderval) {
      epname = name;
      etamin1 = etaminval1;
      etamax1 = etamaxval1;
      etamin2 = etaminval2;
      etamax2 = etamaxval2;
      sumsin = 0;
      sumcos = 0;
      sumsinNoWgt = 0;
      sumcosNoWgt = 0;

      mult = 0;
      order = (double)orderval;
    }
    ~GenPlane() { ; }
    void addParticle(double w, double PtOrEt, double s, double c, double eta) {
      if ((eta >= etamin1 && eta < etamax1) || (etamin2 != etamax2 && eta >= etamin2 && eta < etamax2)) {
        sumsin += w * s;
        sumcos += w * c;
        sumsinNoWgt += s;
        sumcosNoWgt += c;

        sumw += fabs(w);
        sumw2 += w * w;
        sumPtOrEt += PtOrEt;
        sumPtOrEt2 += PtOrEt * PtOrEt;
        ++mult;
      }
    }

    double getAngle(double &ang,
                    double &sv,
                    double &cv,
                    double &svNoWgt,
                    double &cvNoWgt,
                    double &w,
                    double &w2,
                    double &PtOrEt,
                    double &PtOrEt2,
                    uint &epmult) {
      ang = -10;
      sv = sumsin;
      cv = sumcos;
      svNoWgt = sumsinNoWgt;
      cvNoWgt = sumcosNoWgt;
      w = sumw;
      w2 = sumw2;
      PtOrEt = sumPtOrEt;
      PtOrEt2 = sumPtOrEt2;
      epmult = mult;
      double q = sv * sv + cv * cv;
      if (q > 0)
        ang = atan2(sv, cv) / order;
      return ang;
    }
    void reset() {
      sumsin = 0;
      sumcos = 0;
      sumsinNoWgt = 0;
      sumcosNoWgt = 0;
      sumw = 0;
      sumw2 = 0;
      mult = 0;
      sumPtOrEt = 0;
      sumPtOrEt2 = 0;
    }

  private:
    string epname;
    double etamin1;
    double etamax1;

    double etamin2;
    double etamax2;
    double sumsin;
    double sumcos;
    double sumsinNoWgt;
    double sumcosNoWgt;
    uint mult;
    double sumw;
    double sumw2;
    double sumPtOrEt;
    double sumPtOrEt2;
    double order;
  };
}  // namespace hi

class EvtPlaneProducer : public edm::stream::EDProducer<> {
public:
  explicit EvtPlaneProducer(const edm::ParameterSet &);
  ~EvtPlaneProducer() override;

private:
  GenPlane *rp[NumEPNames];

  void produce(edm::Event &, const edm::EventSetup &) override;

  // ----------member data ---------------------------
  EPCuts cuts_;

  std::string centralityVariable_;
  std::string centralityLabel_;
  std::string centralityMC_;

  edm::InputTag centralityBinTag_;
  edm::EDGetTokenT<int> centralityBinToken_;

  edm::InputTag vertexTag_;
  edm::EDGetTokenT<std::vector<reco::Vertex>> vertexToken_;
  edm::Handle<std::vector<reco::Vertex>> vertex_;

  edm::InputTag caloTag_;
  edm::EDGetTokenT<CaloTowerCollection> caloToken_;
  edm::Handle<CaloTowerCollection> caloCollection_;
  edm::EDGetTokenT<reco::PFCandidateCollection> caloTokenPF_;

  edm::InputTag castorTag_;
  edm::EDGetTokenT<std::vector<reco::CastorTower>> castorToken_;
  edm::Handle<std::vector<reco::CastorTower>> castorCollection_;

  edm::InputTag trackTag_;
  edm::EDGetTokenT<reco::TrackCollection> trackToken_;
  edm::InputTag losttrackTag_;
  edm::Handle<reco::TrackCollection> trackCollection_;
  bool bStrack_packedPFCandidates_;
  bool bScalo_particleFlow_;
  edm::EDGetTokenT<pat::PackedCandidateCollection> packedToken_;
  edm::EDGetTokenT<pat::PackedCandidateCollection> lostToken_;

  edm::InputTag chi2MapTag_;
  edm::EDGetTokenT<edm::ValueMap<float>> chi2MapToken_;
  edm::InputTag chi2MapLostTag_;
  edm::EDGetTokenT<edm::ValueMap<float>> chi2MapLostToken_;

  bool loadDB_;
  double minet_;
  double maxet_;
  double minpt_;
  double maxpt_;
  int flatnvtxbins_;
  double flatminvtx_;
  double flatdelvtx_;
  double dzdzerror_;
  double d0d0error_;
  double pterror_;
  double chi2perlayer_;
  double dzerr_;
  double dzdzerror_pix_;
  double chi2_;
  int nhitsValid_;
  int FlatOrder_;
  int NumFlatBins_;
  double nCentBins_;
  double caloCentRef_;
  double caloCentRefWidth_;
  int CentBinCompression_;
  int cutEra_;
  HiEvtPlaneFlatten *flat[NumEPNames];
  TrackStructure track_;

  edm::ESWatcher<HeavyIonRcd> hiWatcher_;
  edm::ESWatcher<HeavyIonRPRcd> hirpWatcher_;

  void fillHF(const TrackStructure &track, double vz, int bin) {
    double minet = minet_;
    double maxet = maxet_;
    for (int i = 0; i < NumEPNames; i++) {
      if (EPDet[i] != HF)
        continue;
      if (minet_ < 0)
        minet = minTransverse[i];
      if (maxet_ < 0)
        maxet = maxTransverse[i];
      if (track.et < minet)
        continue;
      if (track.et > maxet)
        continue;
      if (not passEta(track.eta, i))
        continue;
      double w = track.et;
      if (loadDB_)
        w = track.et * flat[i]->etScale(vz, bin);
      if (EPOrder[i] == 1) {
        if (MomConsWeight[i][0] == 'y' && loadDB_) {
          w = flat[i]->getW(track.et, vz, bin);
        }
      }
      rp[i]->addParticle(w, track.et, sin(EPOrder[i] * track.phi), cos(EPOrder[i] * track.phi), track.eta);
    }
  };

  void fillCastor(const TrackStructure &track, double vz, int bin) {
    double minet = minet_;
    double maxet = maxet_;
    for (int i = 0; i < NumEPNames; i++) {
      if (EPDet[i] == Castor) {
        if (minet_ < 0)
          minet = minTransverse[i];
        if (maxet_ < 0)
          maxet = maxTransverse[i];
        if (track.et < minet)
          continue;
        if (track.et > maxet)
          continue;
        if (not passEta(track.eta, i))
          continue;
        double w = track.et;
        if (EPOrder[i] == 1) {
          if (MomConsWeight[i][0] == 'y' && loadDB_) {
            w = flat[i]->getW(track.et, vz, bin);
          }
        }
        rp[i]->addParticle(w, track.et, sin(EPOrder[i] * track.phi), cos(EPOrder[i] * track.phi), track.eta);
      }
    }
  }

  bool passEta(float eta, int i) {
    if (EPEtaMin2[i] == EPEtaMax2[i]) {
      if (eta < EPEtaMin1[i])
        return false;
      if (eta > EPEtaMax1[i])
        return false;
    } else {
      if (eta < EPEtaMin1[i])
        return false;
      if (eta > EPEtaMax2[i])
        return false;
      if (eta > EPEtaMax1[i] && eta < EPEtaMin2[i])
        return false;
    }
    return true;
  }

  void fillTracker(const TrackStructure &track, double vz, int bin) {
    double minpt = minpt_;
    double maxpt = maxpt_;
    for (int i = 0; i < NumEPNames; i++) {
      if (EPDet[i] == Tracker) {
        if (minpt_ < 0)
          minpt = minTransverse[i];
        if (maxpt_ < 0)
          maxpt = maxTransverse[i];
        if (track.pt < minpt)
          continue;
        if (track.pt > maxpt)
          continue;
        if (not passEta(track.eta, i))
          continue;
        double w = track.pt;
        if (w > 2.5)
          w = 2.0;  //v2 starts decreasing above ~2.5 GeV/c
        if (EPOrder[i] == 1) {
          if (MomConsWeight[i][0] == 'y' && loadDB_) {
            w = flat[i]->getW(track.pt, vz, bin);
          }
        }
        rp[i]->addParticle(w, track.pt, sin(EPOrder[i] * track.phi), cos(EPOrder[i] * track.phi), track.eta);
      }
    }
  };
};

EvtPlaneProducer::EvtPlaneProducer(const edm::ParameterSet &iConfig)
    : centralityVariable_(iConfig.getParameter<std::string>("centralityVariable")),
      centralityBinTag_(iConfig.getParameter<edm::InputTag>("centralityBinTag")),
      vertexTag_(iConfig.getParameter<edm::InputTag>("vertexTag")),
      caloTag_(iConfig.getParameter<edm::InputTag>("caloTag")),
      castorTag_(iConfig.getParameter<edm::InputTag>("castorTag")),
      trackTag_(iConfig.getParameter<edm::InputTag>("trackTag")),
      losttrackTag_(iConfig.getParameter<edm::InputTag>("lostTag")),
      chi2MapTag_(iConfig.getParameter<edm::InputTag>("chi2MapTag")),
      chi2MapLostTag_(iConfig.getParameter<edm::InputTag>("chi2MapLostTag")),
      loadDB_(iConfig.getParameter<bool>("loadDB")),
      minet_(iConfig.getParameter<double>("minet")),
      maxet_(iConfig.getParameter<double>("maxet")),
      minpt_(iConfig.getParameter<double>("minpt")),
      maxpt_(iConfig.getParameter<double>("maxpt")),
      flatnvtxbins_(iConfig.getParameter<int>("flatnvtxbins")),
      flatminvtx_(iConfig.getParameter<double>("flatminvtx")),
      flatdelvtx_(iConfig.getParameter<double>("flatdelvtx")),
      dzdzerror_(iConfig.getParameter<double>("dzdzerror")),
      d0d0error_(iConfig.getParameter<double>("d0d0error")),
      pterror_(iConfig.getParameter<double>("pterror")),
      chi2perlayer_(iConfig.getParameter<double>("chi2perlayer")),
      dzdzerror_pix_(iConfig.getParameter<double>("dzdzerror_pix")),
      chi2_(iConfig.getParameter<double>("chi2")),
      nhitsValid_(iConfig.getParameter<int>("nhitsValid")),
      FlatOrder_(iConfig.getParameter<int>("FlatOrder")),
      NumFlatBins_(iConfig.getParameter<int>("NumFlatBins")),
      caloCentRef_(iConfig.getParameter<double>("caloCentRef")),
      caloCentRefWidth_(iConfig.getParameter<double>("caloCentRefWidth")),
      CentBinCompression_(iConfig.getParameter<int>("CentBinCompression")),
      cutEra_(iConfig.getParameter<int>("cutEra"))

{
  if (cutEra_ > 3)
    throw edm::Exception(edm::errors::Configuration) << "wrong range in cutEra parameter";
  cuts_ = EPCuts(
      static_cast<EP_ERA>(cutEra_), pterror_, dzdzerror_, d0d0error_, chi2perlayer_, dzdzerror_pix_, chi2_, nhitsValid_);
  nCentBins_ = 200.;

  if (iConfig.exists("nonDefaultGlauberModel")) {
    centralityMC_ = iConfig.getParameter<std::string>("nonDefaultGlauberModel");
  }
  centralityLabel_ = centralityVariable_ + centralityMC_;

  centralityBinToken_ = consumes<int>(centralityBinTag_);

  vertexToken_ = consumes<std::vector<reco::Vertex>>(vertexTag_);

  bStrack_packedPFCandidates_ = (trackTag_.label().find("packedPFCandidates") != std::string::npos);
  bScalo_particleFlow_ = (caloTag_.label().find("particleFlow") != std::string::npos);
  if (bStrack_packedPFCandidates_) {
    packedToken_ = consumes<pat::PackedCandidateCollection>(trackTag_);
    lostToken_ = consumes<pat::PackedCandidateCollection>(losttrackTag_);
    chi2MapToken_ = consumes<edm::ValueMap<float>>(chi2MapTag_);
    chi2MapLostToken_ = consumes<edm::ValueMap<float>>(chi2MapLostTag_);

  } else {
    if (bScalo_particleFlow_) {
      caloTokenPF_ = consumes<reco::PFCandidateCollection>(caloTag_);
    } else {
      caloToken_ = consumes<CaloTowerCollection>(caloTag_);
    }
    castorToken_ = consumes<std::vector<reco::CastorTower>>(castorTag_);
    trackToken_ = consumes<reco::TrackCollection>(trackTag_);
  }

  produces<reco::EvtPlaneCollection>();
  for (int i = 0; i < NumEPNames; i++) {
    rp[i] = new GenPlane(EPNames[i], EPEtaMin1[i], EPEtaMax1[i], EPEtaMin2[i], EPEtaMax2[i], EPOrder[i]);
  }
  for (int i = 0; i < NumEPNames; i++) {
    flat[i] = new HiEvtPlaneFlatten();
    flat[i]->init(FlatOrder_, NumFlatBins_, flatnvtxbins_, flatminvtx_, flatdelvtx_, EPNames[i], EPOrder[i]);
  }
}

EvtPlaneProducer::~EvtPlaneProducer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  for (int i = 0; i < NumEPNames; i++) {
    delete flat[i];
  }
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void EvtPlaneProducer::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  using namespace edm;
  using namespace std;
  using namespace reco;
  if (hiWatcher_.check(iSetup)) {
    //
    //Get Size of Centrality Table
    //
    edm::ESHandle<CentralityTable> centDB_;
    iSetup.get<HeavyIonRcd>().get(centralityLabel_, centDB_);
    nCentBins_ = centDB_->m_table.size();
    for (int i = 0; i < NumEPNames; i++) {
      if (caloCentRef_ > 0) {
        int minbin = (caloCentRef_ - caloCentRefWidth_ / 2.) * nCentBins_ / 100.;
        int maxbin = (caloCentRef_ + caloCentRefWidth_ / 2.) * nCentBins_ / 100.;
        minbin /= CentBinCompression_;
        maxbin /= CentBinCompression_;
        if (minbin > 0 && maxbin >= minbin) {
          if (EPDet[i] == HF || EPDet[i] == Castor)
            flat[i]->setCaloCentRefBins(minbin, maxbin);
        }
      }
    }
  }
  //
  //Get flattening parameter file.
  //
  if (loadDB_ && hirpWatcher_.check(iSetup)) {
    edm::ESHandle<RPFlatParams> flatparmsDB_;
    iSetup.get<HeavyIonRPRcd>().get(flatparmsDB_);
    LoadEPDB db(flatparmsDB_, flat);
    if (!db.IsSuccess()) {
      loadDB_ = kFALSE;
    }
  }
  //
  //Get Centrality
  //
  int bin = 0;
  int cbin = 0;
  if (loadDB_) {
    cbin = iEvent.get(centralityBinToken_);
    bin = cbin / CentBinCompression_;
  }
  //
  //Get Vertex
  //
  //best vertex
  const reco::Vertex &vtx = iEvent.get(vertexToken_)[0];
  double bestvz = vtx.z();
  double bestvx = vtx.x();
  double bestvy = vtx.y();
  double bestvzError = vtx.zError();
  math::XYZPoint bestvtx(bestvx, bestvy, bestvz);
  math::Error<3>::type vtx_cov = vtx.covariance();

  for (int i = 0; i < NumEPNames; i++)
    rp[i]->reset();
  edm::Handle<edm::ValueMap<float>> chi2Map;
  edm::Handle<pat::PackedCandidateCollection> cands;
  edm::Handle<reco::PFCandidateCollection> calocands;
  if (bStrack_packedPFCandidates_) {
    for (int idx = 1; idx < 3; idx++) {
      if (idx == 1) {
        iEvent.getByToken(packedToken_, cands);
        iEvent.getByToken(chi2MapToken_, chi2Map);
      }
      if (idx == 2) {
        iEvent.getByToken(lostToken_, cands);
        iEvent.getByToken(chi2MapLostToken_, chi2Map);
      }
      for (unsigned int i = 0, n = cands->size(); i < n; ++i) {
        track_ = {};
        track_.centbin = cbin;
        const pat::PackedCandidate &pf = (*cands)[i];
        track_.et = pf.et();
        track_.eta = pf.eta();
        track_.phi = pf.phi();
        track_.pdgid = pf.pdgId();
        if ((idx == 1) and cuts_.isGoodHF(track_)) {
          fillHF(track_, bestvz, bin);
        }
        if (!pf.hasTrackDetails())
          continue;
        const reco::Track &trk = pf.pseudoTrack();
        track_.highPurity = pf.trackHighPurity();
        track_.charge = trk.charge();
        if (!track_.highPurity || track_.charge == 0)
          continue;
        track_.collection = idx;
        track_.eta = trk.eta();
        track_.phi = trk.phi();
        track_.pt = trk.pt();
        track_.ptError = trk.ptError();
        track_.numberOfValidHits = trk.numberOfValidHits();
        track_.algos = trk.algo();
        track_.dz = std::abs(trk.dz(bestvtx));
        track_.dxy = std::abs(trk.dxy(bestvtx));
        track_.dzError = std::hypot(trk.dzError(), bestvzError);
        track_.dxyError = trk.dxyError(bestvtx, vtx_cov);
        track_.dzSig = track_.dz / track_.dzError;
        track_.dxySig = track_.dxy / track_.dxyError;
        const reco::HitPattern &hit_pattern = trk.hitPattern();
        track_.normalizedChi2 = (*chi2Map)[pat::PackedCandidateRef(cands, i)];
        track_.chi2layer = (*chi2Map)[pat::PackedCandidateRef(cands, i)] / hit_pattern.trackerLayersWithMeasurement();
        if (cuts_.isGoodTrack(track_)) {
          fillTracker(track_, bestvz, bin);
        }
      }
    }
  } else {
    //calorimetry part
    if (bScalo_particleFlow_) {
      iEvent.getByToken(caloTokenPF_, calocands);
      for (unsigned int i = 0, n = calocands->size(); i < n; ++i) {
        track_ = {};
        track_.centbin = cbin;
        const reco::PFCandidate &pf = (*calocands)[i];
        track_.et = pf.et();
        track_.eta = pf.eta();
        track_.phi = pf.phi();
        track_.pdgid = pf.pdgId();
        if (cuts_.isGoodHF(track_)) {
          fillHF(track_, bestvz, bin);
        }
      }
    } else {
      iEvent.getByToken(caloToken_, caloCollection_);
      for (const auto &tower : *caloCollection_) {
        track_.eta = tower.eta();
        track_.phi = tower.phi();
        track_.et = tower.emEt() + tower.hadEt();
        track_.pdgid = 1;
        if (cuts_.isGoodHF(track_))
          fillHF(track_, bestvz, bin);
      }
    }

    //Castor part
    iEvent.getByToken(castorToken_, castorCollection_);
    for (const auto &tower : *castorCollection_) {
      track_.eta = tower.eta();
      track_.phi = tower.phi();
      track_.et = tower.et();
      track_.pdgid = 1;
      if (cuts_.isGoodCastor(track_))
        fillCastor(track_, bestvz, bin);
    }
    //Tracking part
    iEvent.getByToken(trackToken_, trackCollection_);
    for (const auto &trk : *trackCollection_) {
      track_.highPurity = trk.quality(reco::TrackBase::highPurity);
      track_.charge = trk.charge();
      if (!track_.highPurity || track_.charge == 0)
        continue;
      track_.centbin = cbin;
      track_.collection = 0;
      track_.eta = trk.eta();
      track_.phi = trk.phi();
      track_.pt = trk.pt();
      track_.ptError = trk.ptError();
      track_.numberOfValidHits = trk.numberOfValidHits();
      track_.algos = trk.algo();
      track_.dz = std::abs(trk.dz(bestvtx));
      track_.dxy = std::abs(trk.dxy(bestvtx));
      track_.dzError = std::hypot(trk.dzError(), bestvzError);
      track_.dxyError = trk.dxyError(bestvtx, vtx_cov);
      track_.dzSig = track_.dz / track_.dzError;
      track_.dxySig = track_.dxy / track_.dxyError;
      track_.normalizedChi2 = trk.normalizedChi2();
      track_.chi2layer = track_.normalizedChi2 / trk.hitPattern().trackerLayersWithMeasurement();
      if (cuts_.isGoodTrack(track_))
        fillTracker(track_, bestvz, bin);
    }
  }

  auto evtplaneOutput = std::make_unique<EvtPlaneCollection>();

  double ang = -10;
  double sv = 0;
  double cv = 0;
  double svNoWgt = 0;
  double cvNoWgt = 0;

  double wv = 0;
  double wv2 = 0;
  double pe = 0;
  double pe2 = 0;
  uint epmult = 0;

  for (int i = 0; i < NumEPNames; i++) {
    rp[i]->getAngle(ang, sv, cv, svNoWgt, cvNoWgt, wv, wv2, pe, pe2, epmult);
    evtplaneOutput->push_back(EvtPlane(i, 0, ang, sv, cv, wv, wv2, pe, pe2, epmult));
    evtplaneOutput->back().addLevel(3, 0., svNoWgt, cvNoWgt);
  }

  iEvent.put(std::move(evtplaneOutput));
}

//define this as a plug-in
DEFINE_FWK_MODULE(EvtPlaneProducer);

// -*- C++ -*-
//
// Package:    Alignment/OfflineValidation
// Class:      GeneralPurposeTrackAnalyzer
//
/**\class GeneralPurposeTrackAnalyzer GeneralPurposeTrackAnalyzer.cc Alignment/OfflineValidation/plugins/GeneralPurposeTrackAnalyzer.cc

*/
//
// Original Author:  Marco Musich
//         Created:  Mon, 13 Jun 2016 15:07:11 GMT
//
//

// ROOT includes files

#include "TMath.h"
#include "TFile.h"
#include "TH1D.h"
#include "TH1I.h"
#include "TH2D.h"
#include "TProfile.h"

// system includes files

#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <boost/range/adaptor/indexed.hpp>

// user include files
#include "CommonTools/TrackerMap/interface/TrackerMap.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CondCore/SiPixelPlugins/interface/SiPixelPayloadInspectorHelper.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/SiStripObjects/interface/SiStripLatency.h"
#include "DQM/SiPixelPhase1Common/interface/SiPixelCoordinates.h"
#include "DQM/TrackerRemapper/interface/Phase1PixelMaps.h"
#include "DQM/TrackerRemapper/interface/Phase1PixelROCMaps.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerMultiRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"

// toggle to enable debugging
#define DEBUG 0

const int kBPIX = PixelSubdetector::PixelBarrel;
const int kFPIX = PixelSubdetector::PixelEndcap;

class GeneralPurposeTrackAnalyzer : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
public:
  GeneralPurposeTrackAnalyzer(const edm::ParameterSet &pset)
      : geomToken_(esConsumes()),
        magFieldToken_(esConsumes<edm::Transition::BeginRun>()),
        geomTokenBR_(esConsumes<edm::Transition::BeginRun>()),
        trackerTopologyTokenBR_(esConsumes<edm::Transition::BeginRun>()),
        siPixelFedCablingMapTokenBR_(esConsumes<edm::Transition::BeginRun>()) {
    doLatencyAnalysis_ = pset.getParameter<bool>("doLatencyAnalysis");
    if (doLatencyAnalysis_) {
      latencyToken_ = esConsumes<edm::Transition::BeginRun>();
    }

    usesResource(TFileService::kSharedResource);

    TkTag_ = pset.getParameter<edm::InputTag>("TkTag");
    theTrackCollectionToken_ = consumes<reco::TrackCollection>(TkTag_);

    TriggerResultsTag_ = pset.getParameter<edm::InputTag>("TriggerResultsTag");
    hltresultsToken_ = consumes<edm::TriggerResults>(TriggerResultsTag_);

    BeamSpotTag_ = pset.getParameter<edm::InputTag>("BeamSpotTag");
    beamspotToken_ = consumes<reco::BeamSpot>(BeamSpotTag_);

    VerticesTag_ = pset.getParameter<edm::InputTag>("VerticesTag");
    vertexToken_ = consumes<reco::VertexCollection>(VerticesTag_);

    isCosmics_ = pset.getParameter<bool>("isCosmics");

    pmap = std::make_unique<TrackerMap>("Pixel");
    pmap->onlyPixel(true);
    pmap->setTitle("Pixel Hit entries");
    pmap->setPalette(1);

    tmap = std::make_unique<TrackerMap>("Strip");
    tmap->setTitle("Strip Hit entries");
    tmap->setPalette(1);

    pixelmap = std::make_unique<Phase1PixelMaps>("COLZ0 L");
    pixelmap->bookBarrelHistograms("entriesBarrel", "# hits", "# pixel hits");
    pixelmap->bookForwardHistograms("entriesForward", "# hits", "# pixel hits");

    pixelrocsmap_ = std::make_unique<Phase1PixelROCMaps>("");
  }

  ~GeneralPurposeTrackAnalyzer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions &);

  template <class OBJECT_TYPE>
  int index(const std::vector<OBJECT_TYPE *> &vec, const TString &name) {
    for (const auto &iter : vec | boost::adaptors::indexed(0)) {
      if (iter.value() && iter.value()->GetName() == name) {
        return iter.index();
      }
    }
    edm::LogError("GeneralPurposeTrackAnalyzer") << "@SUB=GeneralPurposeTrackAnalyzer::index"
                                                 << " could not find " << name;
    return -1;
  }

  template <typename T, typename... Args>
  T *book(const Args &...args) const {
    T *t = fs->make<T>(args...);
    return t;
  }

private:
  // tokens for the event setup
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magFieldToken_;
  edm::ESGetToken<SiStripLatency, SiStripLatencyRcd> latencyToken_;

  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomTokenBR_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopologyTokenBR_;
  const edm::ESGetToken<SiPixelFedCablingMap, SiPixelFedCablingMapRcd> siPixelFedCablingMapTokenBR_;

  edm::ESHandle<MagneticField> magneticField_;

  SiPixelCoordinates coord_;

  edm::Service<TFileService> fs;

  std::unique_ptr<TrackerMap> tmap;
  std::unique_ptr<TrackerMap> pmap;

  std::unique_ptr<Phase1PixelMaps> pixelmap;
  std::unique_ptr<Phase1PixelROCMaps> pixelrocsmap_;

  TH1D *hchi2ndof;
  TH1D *hNtrk;
  TH1D *hNtrkZoom;
  TH1I *htrkQuality;
  TH1I *htrkAlgo;
  TH1I *htrkOriAlgo;
  TH1D *hNhighPurity;
  TH1D *hP;
  TH1D *hPPlus;
  TH1D *hPMinus;
  TH1D *hPt;
  TH1D *hMinPt;
  TH1D *hPtPlus;
  TH1D *hPtMinus;
  TH1D *hHit;
  TH1D *hHit2D;

  TH1D *hHitCountVsXBPix;
  TH1D *hHitCountVsXFPix;
  TH1D *hHitCountVsYBPix;
  TH1D *hHitCountVsYFPix;
  TH1D *hHitCountVsZBPix;
  TH1D *hHitCountVsZFPix;

  TH1D *hHitCountVsThetaBPix;
  TH1D *hHitCountVsPhiBPix;

  TH1D *hHitCountVsThetaFPix;
  TH1D *hHitCountVsPhiFPix;

  TH1D *hHitPlus;
  TH1D *hHitMinus;

  TH1D *hPhp;
  TH1D *hPthp;
  TH1D *hHithp;
  TH1D *hEtahp;
  TH1D *hPhihp;
  TH1D *hchi2ndofhp;
  TH1D *hchi2Probhp;

  TH1D *hCharge;
  TH1D *hQoverP;
  TH1D *hQoverPZoom;
  TH1D *hEta;
  TH1D *hEtaPlus;
  TH1D *hEtaMinus;
  TH1D *hPhi;
  TH1D *hPhiBarrel;
  TH1D *hPhiOverlapPlus;
  TH1D *hPhiOverlapMinus;
  TH1D *hPhiEndcapPlus;
  TH1D *hPhiEndcapMinus;
  TH1D *hPhiPlus;
  TH1D *hPhiMinus;

  TH1D *hDeltaPhi;
  TH1D *hDeltaEta;
  TH1D *hDeltaR;

  TH1D *hvx;
  TH1D *hvy;
  TH1D *hvz;
  TH1D *hd0;
  TH1D *hdz;
  TH1D *hdxy;

  TH2D *hd0PVvsphi;
  TH2D *hd0PVvseta;
  TH2D *hd0PVvspt;

  TH2D *hd0vsphi;
  TH2D *hd0vseta;
  TH2D *hd0vspt;

  TH1D *hnhpxb;
  TH1D *hnhpxe;
  TH1D *hnhTIB;
  TH1D *hnhTID;
  TH1D *hnhTOB;
  TH1D *hnhTEC;

  TH1D *hHitComposition;

  TProfile *pNBpixHitsVsVx;
  TProfile *pNBpixHitsVsVy;
  TProfile *pNBpixHitsVsVz;

  TH1D *hMultCand;

  TH1D *hdxyBS;
  TH1D *hd0BS;
  TH1D *hdzBS;
  TH1D *hdxyPV;
  TH1D *hd0PV;
  TH1D *hdzPV;
  TH1D *hrun;
  TH1D *hlumi;

  std::vector<TH1 *> vTrackHistos_;
  std::vector<TH1 *> vTrackProfiles_;
  std::vector<TH1 *> vTrack2DHistos_;

  TH1D *tksByTrigger_;
  TH1D *evtsByTrigger_;

  TH1D *modeByRun_;
  TH1D *fieldByRun_;

  int ievt;
  int itrks;
  int mode;
  float etaMax_;
  SiPixelPI::phase phase_;

  const TrackerGeometry *trackerGeometry_;

  edm::InputTag TkTag_;
  edm::InputTag TriggerResultsTag_;
  edm::InputTag BeamSpotTag_;
  edm::InputTag VerticesTag_;

  bool isCosmics_;
  bool doLatencyAnalysis_;

  edm::EDGetTokenT<reco::TrackCollection> theTrackCollectionToken_;
  edm::EDGetTokenT<edm::TriggerResults> hltresultsToken_;
  edm::EDGetTokenT<reco::BeamSpot> beamspotToken_;
  edm::EDGetTokenT<reco::VertexCollection> vertexToken_;

  std::map<std::string, std::pair<int, int> > triggerMap_;
  std::map<int, std::pair<int, float> > conditionsMap_;
  std::map<int, std::pair<int, int> > runInfoMap_;

  //*************************************************************
  void analyze(const edm::Event &event, const edm::EventSetup &setup) override
  //*************************************************************
  {
    ievt++;

    // geometry setup
    const TrackerGeometry *theGeometry = &setup.getData(geomToken_);

    edm::Handle<reco::TrackCollection> trackCollection = event.getHandle(theTrackCollectionToken_);
    const reco::TrackCollection tC = *(trackCollection.product());
    itrks += tC.size();

    runInfoMap_[event.run()].first += 1;
    runInfoMap_[event.run()].second += tC.size();

    if (DEBUG) {
      edm::LogInfo("GeneralPurposeTrackAnalyzer") << "Reconstructed " << tC.size() << " tracks" << std::endl;
    }
    //int iCounter=0;

    edm::Handle<edm::TriggerResults> hltresults = event.getHandle(hltresultsToken_);
    if (hltresults.isValid()) {
      const edm::TriggerNames &triggerNames_ = event.triggerNames(*hltresults);
      int ntrigs = hltresults->size();
      //const vector<std::string> &triggernames = triggerNames_.triggerNames();

      for (int itrig = 0; itrig != ntrigs; ++itrig) {
        const std::string &trigName = triggerNames_.triggerName(itrig);
        bool accept = hltresults->accept(itrig);
        if (accept == 1) {
          if (DEBUG) {
            edm::LogInfo("GeneralPurposeTrackAnalyzer")
                << trigName << " " << accept << " ,track size: " << tC.size() << std::endl;
          }
          triggerMap_[trigName].first += 1;
          triggerMap_[trigName].second += tC.size();
          // triggerInfo.push_back(pair <std::string, int> (trigName, accept));
        }
      }
    }

    hrun->Fill(event.run());
    hlumi->Fill(event.luminosityBlock());

    int nHighPurityTracks = 0;

    for (auto track = tC.cbegin(); track != tC.cend(); track++) {
      unsigned int nHit2D = 0;
      std::bitset<16> rocsToMask;
      for (auto iHit = track->recHitsBegin(); iHit != track->recHitsEnd(); ++iHit) {
        if (this->isHit2D(**iHit)) {
          ++nHit2D;
        }
        // rest the ROCs for the map
        rocsToMask.reset();
        const DetId &detId = (*iHit)->geographicalId();
        const GeomDet *geomDet(theGeometry->idToDet(detId));

        const SiPixelRecHit *pixhit = dynamic_cast<const SiPixelRecHit *>(*iHit);

        if (pixhit) {
          if (pixhit->isValid()) {
            unsigned int subid = detId.subdetId();
            int detid_db = detId.rawId();

            // get the cluster
            auto clustp = pixhit->cluster();

            if (clustp.isNull())
              continue;
            auto const &cluster = *clustp;
            int row = cluster.x() - 0.5, col = cluster.y() - 0.5;
            int rocId = coord_.roc(detId, std::make_pair(row, col));

            if (phase_ == SiPixelPI::phase::zero) {
              pmap->fill(detid_db, 1);
            } else if (phase_ == SiPixelPI::phase::one) {
              rocsToMask.set(rocId);
              pixelrocsmap_->fillSelectedRocs(detid_db, rocsToMask, 1);

              if (subid == PixelSubdetector::PixelBarrel) {
                pixelmap->fillBarrelBin("entriesBarrel", detid_db, 1);
              } else {
                pixelmap->fillForwardBin("entriesForward", detid_db, 1);
              }
            }

            LocalPoint lp = (*iHit)->localPosition();
            //LocalError le = (*iHit)->localPositionError();

            GlobalPoint GP = geomDet->surface().toGlobal(lp);

            if ((subid == PixelSubdetector::PixelBarrel) || (subid == PixelSubdetector::PixelEndcap)) {
              // 1 = PXB, 2 = PXF
              if (subid == PixelSubdetector::PixelBarrel) {
                hHitCountVsThetaBPix->Fill(GP.theta());
                hHitCountVsPhiBPix->Fill(GP.phi());

                hHitCountVsZBPix->Fill(GP.z());
                hHitCountVsXBPix->Fill(GP.x());
                hHitCountVsYBPix->Fill(GP.y());

              } else if (subid == PixelSubdetector::PixelEndcap) {
                hHitCountVsThetaFPix->Fill(GP.theta());
                hHitCountVsPhiFPix->Fill(GP.phi());

                hHitCountVsZFPix->Fill(GP.z());
                hHitCountVsXFPix->Fill(GP.x());
                hHitCountVsYFPix->Fill(GP.y());
              }
            }
          }
        } else {
          if ((*iHit)->isValid() && phase_ != SiPixelPI::phase::two) {
            tmap->fill(detId.rawId(), 1);
          }
        }
      }

      hHit2D->Fill(nHit2D);
      hHit->Fill(track->numberOfValidHits());
      hnhpxb->Fill(track->hitPattern().numberOfValidPixelBarrelHits());
      hnhpxe->Fill(track->hitPattern().numberOfValidPixelEndcapHits());
      hnhTIB->Fill(track->hitPattern().numberOfValidStripTIBHits());
      hnhTID->Fill(track->hitPattern().numberOfValidStripTIDHits());
      hnhTOB->Fill(track->hitPattern().numberOfValidStripTOBHits());
      hnhTEC->Fill(track->hitPattern().numberOfValidStripTECHits());

      // fill hit composition histogram
      if (track->hitPattern().numberOfValidPixelBarrelHits() != 0) {
        hHitComposition->Fill(0., track->hitPattern().numberOfValidPixelBarrelHits());

        pNBpixHitsVsVx->Fill(track->vx(), track->hitPattern().numberOfValidPixelBarrelHits());
        pNBpixHitsVsVy->Fill(track->vy(), track->hitPattern().numberOfValidPixelBarrelHits());
        pNBpixHitsVsVz->Fill(track->vz(), track->hitPattern().numberOfValidPixelBarrelHits());
      }
      if (track->hitPattern().numberOfValidPixelEndcapHits() != 0) {
        hHitComposition->Fill(1., track->hitPattern().numberOfValidPixelEndcapHits());
      }
      if (track->hitPattern().numberOfValidStripTIBHits() != 0) {
        hHitComposition->Fill(2., track->hitPattern().numberOfValidStripTIBHits());
      }
      if (track->hitPattern().numberOfValidStripTIDHits() != 0) {
        hHitComposition->Fill(3., track->hitPattern().numberOfValidStripTIDHits());
      }
      if (track->hitPattern().numberOfValidStripTOBHits() != 0) {
        hHitComposition->Fill(4., track->hitPattern().numberOfValidStripTOBHits());
      }
      if (track->hitPattern().numberOfValidStripTECHits() != 0) {
        hHitComposition->Fill(5., track->hitPattern().numberOfValidStripTECHits());
      }

      hCharge->Fill(track->charge());
      hQoverP->Fill(track->qoverp());
      hQoverPZoom->Fill(track->qoverp());
      hPt->Fill(track->pt());
      hP->Fill(track->p());
      hchi2ndof->Fill(track->normalizedChi2());
      hEta->Fill(track->eta());
      hPhi->Fill(track->phi());

      if (fabs(track->eta()) < 0.8) {
        hPhiBarrel->Fill(track->phi());
      }
      if (track->eta() > 0.8 && track->eta() < 1.4) {
        hPhiOverlapPlus->Fill(track->phi());
      }
      if (track->eta() < -0.8 && track->eta() > -1.4) {
        hPhiOverlapMinus->Fill(track->phi());
      }
      if (track->eta() > 1.4) {
        hPhiEndcapPlus->Fill(track->phi());
      }
      if (track->eta() < -1.4) {
        hPhiEndcapMinus->Fill(track->phi());
      }

      hd0->Fill(track->d0());
      hdz->Fill(track->dz());
      hdxy->Fill(track->dxy());
      hvx->Fill(track->vx());
      hvy->Fill(track->vy());
      hvz->Fill(track->vz());

      htrkAlgo->Fill(static_cast<int>(track->algo()));
      htrkOriAlgo->Fill(static_cast<int>(track->originalAlgo()));

      int myquality = -99;
      if (track->quality(reco::TrackBase::undefQuality)) {
        myquality = -1;
        htrkQuality->Fill(myquality);
      }
      if (track->quality(reco::TrackBase::loose)) {
        myquality = 0;
        htrkQuality->Fill(myquality);
      }
      if (track->quality(reco::TrackBase::tight)) {
        myquality = 1;
        htrkQuality->Fill(myquality);
      }
      if (track->quality(reco::TrackBase::highPurity) && (!isCosmics_)) {
        myquality = 2;
        htrkQuality->Fill(myquality);
        hPhp->Fill(track->p());
        hPthp->Fill(track->pt());
        hHithp->Fill(track->numberOfValidHits());
        hEtahp->Fill(track->eta());
        hPhihp->Fill(track->phi());
        hchi2ndofhp->Fill(track->normalizedChi2());
        hchi2Probhp->Fill(TMath::Prob(track->chi2(), track->ndof()));
        nHighPurityTracks++;
      }
      if (track->quality(reco::TrackBase::confirmed)) {
        myquality = 3;
        htrkQuality->Fill(myquality);
      }
      if (track->quality(reco::TrackBase::goodIterative)) {
        myquality = 4;
        htrkQuality->Fill(myquality);
      }

      // Fill 1D track histos
      static const int etaindex = this->index(vTrackHistos_, "h_tracketa");
      vTrackHistos_[etaindex]->Fill(track->eta());
      static const int phiindex = this->index(vTrackHistos_, "h_trackphi");
      vTrackHistos_[phiindex]->Fill(track->phi());
      static const int numOfValidHitsindex = this->index(vTrackHistos_, "h_trackNumberOfValidHits");
      vTrackHistos_[numOfValidHitsindex]->Fill(track->numberOfValidHits());
      static const int numOfLostHitsindex = this->index(vTrackHistos_, "h_trackNumberOfLostHits");
      vTrackHistos_[numOfLostHitsindex]->Fill(track->numberOfLostHits());

      GlobalPoint gPoint(track->vx(), track->vy(), track->vz());
      double theLocalMagFieldInInverseGeV = magneticField_->inInverseGeV(gPoint).z();
      double kappa = -track->charge() * theLocalMagFieldInInverseGeV / track->pt();

      static const int kappaindex = this->index(vTrackHistos_, "h_curvature");
      vTrackHistos_[kappaindex]->Fill(kappa);
      static const int kappaposindex = this->index(vTrackHistos_, "h_curvature_pos");
      if (track->charge() > 0)
        vTrackHistos_[kappaposindex]->Fill(fabs(kappa));
      static const int kappanegindex = this->index(vTrackHistos_, "h_curvature_neg");
      if (track->charge() < 0)
        vTrackHistos_[kappanegindex]->Fill(fabs(kappa));

      double chi2Prob = TMath::Prob(track->chi2(), track->ndof());
      double normchi2 = track->normalizedChi2();

      static const int normchi2index = this->index(vTrackHistos_, "h_normchi2");
      vTrackHistos_[normchi2index]->Fill(normchi2);
      static const int chi2index = this->index(vTrackHistos_, "h_chi2");
      vTrackHistos_[chi2index]->Fill(track->chi2());
      static const int chi2Probindex = this->index(vTrackHistos_, "h_chi2Prob");
      vTrackHistos_[chi2Probindex]->Fill(chi2Prob);
      static const int ptindex = this->index(vTrackHistos_, "h_pt");
      static const int pt2index = this->index(vTrackHistos_, "h_ptrebin");
      vTrackHistos_[ptindex]->Fill(track->pt());
      vTrackHistos_[pt2index]->Fill(track->pt());
      if (track->ptError() != 0.) {
        static const int ptResolutionindex = this->index(vTrackHistos_, "h_ptResolution");
        vTrackHistos_[ptResolutionindex]->Fill(track->ptError() / track->pt());
      }
      // Fill track profiles
      static const int d0phiindex = this->index(vTrackProfiles_, "p_d0_vs_phi");
      vTrackProfiles_[d0phiindex]->Fill(track->phi(), track->d0());
      static const int dzphiindex = this->index(vTrackProfiles_, "p_dz_vs_phi");
      vTrackProfiles_[dzphiindex]->Fill(track->phi(), track->dz());
      static const int d0etaindex = this->index(vTrackProfiles_, "p_d0_vs_eta");
      vTrackProfiles_[d0etaindex]->Fill(track->eta(), track->d0());
      static const int dzetaindex = this->index(vTrackProfiles_, "p_dz_vs_eta");
      vTrackProfiles_[dzetaindex]->Fill(track->eta(), track->dz());
      static const int chiProbphiindex = this->index(vTrackProfiles_, "p_chi2Prob_vs_phi");
      vTrackProfiles_[chiProbphiindex]->Fill(track->phi(), chi2Prob);
      static const int chiProbabsd0index = this->index(vTrackProfiles_, "p_chi2Prob_vs_d0");
      vTrackProfiles_[chiProbabsd0index]->Fill(fabs(track->d0()), chi2Prob);
      static const int chiProbabsdzindex = this->index(vTrackProfiles_, "p_chi2Prob_vs_dz");
      vTrackProfiles_[chiProbabsdzindex]->Fill(track->dz(), chi2Prob);
      static const int chiphiindex = this->index(vTrackProfiles_, "p_chi2_vs_phi");
      vTrackProfiles_[chiphiindex]->Fill(track->phi(), track->chi2());
      static const int normchiphiindex = this->index(vTrackProfiles_, "p_normchi2_vs_phi");
      vTrackProfiles_[normchiphiindex]->Fill(track->phi(), normchi2);
      static const int chietaindex = this->index(vTrackProfiles_, "p_chi2_vs_eta");
      vTrackProfiles_[chietaindex]->Fill(track->eta(), track->chi2());
      static const int normchiptindex = this->index(vTrackProfiles_, "p_normchi2_vs_pt");
      vTrackProfiles_[normchiptindex]->Fill(track->pt(), normchi2);
      static const int normchipindex = this->index(vTrackProfiles_, "p_normchi2_vs_p");
      vTrackProfiles_[normchipindex]->Fill(track->p(), normchi2);
      static const int chiProbetaindex = this->index(vTrackProfiles_, "p_chi2Prob_vs_eta");
      vTrackProfiles_[chiProbetaindex]->Fill(track->eta(), chi2Prob);
      static const int normchietaindex = this->index(vTrackProfiles_, "p_normchi2_vs_eta");
      vTrackProfiles_[normchietaindex]->Fill(track->eta(), normchi2);
      static const int kappaphiindex = this->index(vTrackProfiles_, "p_kappa_vs_phi");
      vTrackProfiles_[kappaphiindex]->Fill(track->phi(), kappa);
      static const int kappaetaindex = this->index(vTrackProfiles_, "p_kappa_vs_eta");
      vTrackProfiles_[kappaetaindex]->Fill(track->eta(), kappa);
      static const int ptResphiindex = this->index(vTrackProfiles_, "p_ptResolution_vs_phi");
      vTrackProfiles_[ptResphiindex]->Fill(track->phi(), track->ptError() / track->pt());
      static const int ptResetaindex = this->index(vTrackProfiles_, "p_ptResolution_vs_eta");
      vTrackProfiles_[ptResetaindex]->Fill(track->eta(), track->ptError() / track->pt());

      // Fill 2D track histos
      static const int etaphiindex_2d = this->index(vTrack2DHistos_, "h2_phi_vs_eta");
      vTrack2DHistos_[etaphiindex_2d]->Fill(track->eta(), track->phi());
      static const int d0phiindex_2d = this->index(vTrack2DHistos_, "h2_d0_vs_phi");
      vTrack2DHistos_[d0phiindex_2d]->Fill(track->phi(), track->d0());
      static const int dzphiindex_2d = this->index(vTrack2DHistos_, "h2_dz_vs_phi");
      vTrack2DHistos_[dzphiindex_2d]->Fill(track->phi(), track->dz());
      static const int d0etaindex_2d = this->index(vTrack2DHistos_, "h2_d0_vs_eta");
      vTrack2DHistos_[d0etaindex_2d]->Fill(track->eta(), track->d0());
      static const int dzetaindex_2d = this->index(vTrack2DHistos_, "h2_dz_vs_eta");
      vTrack2DHistos_[dzetaindex_2d]->Fill(track->eta(), track->dz());
      static const int chiphiindex_2d = this->index(vTrack2DHistos_, "h2_chi2_vs_phi");
      vTrack2DHistos_[chiphiindex_2d]->Fill(track->phi(), track->chi2());
      static const int chiProbphiindex_2d = this->index(vTrack2DHistos_, "h2_chi2Prob_vs_phi");
      vTrack2DHistos_[chiProbphiindex_2d]->Fill(track->phi(), chi2Prob);
      static const int chiProbabsd0index_2d = this->index(vTrack2DHistos_, "h2_chi2Prob_vs_d0");
      vTrack2DHistos_[chiProbabsd0index_2d]->Fill(fabs(track->d0()), chi2Prob);
      static const int normchiphiindex_2d = this->index(vTrack2DHistos_, "h2_normchi2_vs_phi");
      vTrack2DHistos_[normchiphiindex_2d]->Fill(track->phi(), normchi2);
      static const int chietaindex_2d = this->index(vTrack2DHistos_, "h2_chi2_vs_eta");
      vTrack2DHistos_[chietaindex_2d]->Fill(track->eta(), track->chi2());
      static const int chiProbetaindex_2d = this->index(vTrack2DHistos_, "h2_chi2Prob_vs_eta");
      vTrack2DHistos_[chiProbetaindex_2d]->Fill(track->eta(), chi2Prob);
      static const int normchietaindex_2d = this->index(vTrack2DHistos_, "h2_normchi2_vs_eta");
      vTrack2DHistos_[normchietaindex_2d]->Fill(track->eta(), normchi2);
      static const int kappaphiindex_2d = this->index(vTrack2DHistos_, "h2_kappa_vs_phi");
      vTrack2DHistos_[kappaphiindex_2d]->Fill(track->phi(), kappa);
      static const int kappaetaindex_2d = this->index(vTrack2DHistos_, "h2_kappa_vs_eta");
      vTrack2DHistos_[kappaetaindex_2d]->Fill(track->eta(), kappa);
      static const int normchi2kappa_2d = this->index(vTrack2DHistos_, "h2_normchi2_vs_kappa");
      vTrack2DHistos_[normchi2kappa_2d]->Fill(normchi2, kappa);

      //dxy with respect to the beamspot
      reco::BeamSpot beamSpot;
      edm::Handle<reco::BeamSpot> beamSpotHandle = event.getHandle(beamspotToken_);
      if (beamSpotHandle.isValid()) {
        beamSpot = *beamSpotHandle;
        math::XYZPoint point(beamSpot.x0(), beamSpot.y0(), beamSpot.z0());
        double dxy = track->dxy(point);
        double dz = track->dz(point);
        hdxyBS->Fill(dxy);
        hd0BS->Fill(-dxy);
        hdzBS->Fill(dz);
      }

      //dxy with respect to the primary vertex
      reco::Vertex pvtx;
      edm::Handle<reco::VertexCollection> vertexHandle = event.getHandle(vertexToken_);
      double mindxy = 100.;
      double dz = 100;
      if (vertexHandle.isValid() && !isCosmics_) {
        for (auto pvtx = vertexHandle->cbegin(); pvtx != vertexHandle->cend(); ++pvtx) {
          math::XYZPoint mypoint(pvtx->x(), pvtx->y(), pvtx->z());
          if (abs(mindxy) > abs(track->dxy(mypoint))) {
            mindxy = track->dxy(mypoint);
            dz = track->dz(mypoint);
          }
        }

        hdxyPV->Fill(mindxy);
        hd0PV->Fill(-mindxy);
        hdzPV->Fill(dz);

        hd0PVvsphi->Fill(track->phi(), -mindxy);
        hd0PVvseta->Fill(track->eta(), -mindxy);
        hd0PVvspt->Fill(track->pt(), -mindxy);

      } else {
        hdxyPV->Fill(100);
        hd0PV->Fill(100);
        hdzPV->Fill(100);

        hd0vsphi->Fill(track->phi(), -track->dxy());
        hd0vseta->Fill(track->eta(), -track->dxy());
        hd0vspt->Fill(track->pt(), -track->dxy());
      }

      if (DEBUG) {
        edm::LogInfo("GeneralPurposeTrackAnalyzer") << "end of track loop" << std::endl;
      }
    }

    hNtrk->Fill(tC.size());
    hNtrkZoom->Fill(tC.size());
    hNhighPurity->Fill(nHighPurityTracks);

    if (DEBUG) {
      edm::LogInfo("GeneralPurposeTrackAnalyzer") << "end of analysis" << std::endl;
    }
  }

  //*************************************************************
  void beginRun(edm::Run const &run, edm::EventSetup const &setup) override
  //*************************************************************
  {
    // initialize runInfoMap_
    runInfoMap_[run.run()].first = 0;
    runInfoMap_[run.run()].second = 0;

    // Magnetic Field setup
    magneticField_ = setup.getHandle(magFieldToken_);
    float B_ = magneticField_.product()->inTesla(GlobalPoint(0, 0, 0)).mag();

    if (DEBUG) {
      edm::LogInfo("GeneralPurposeTrackAnalyzer")
          << "run number:" << run.run() << " magnetic field: " << B_ << " [T]" << std::endl;
    }

    const TrackerGeometry *trackerGeometry = &setup.getData(geomTokenBR_);
    if (trackerGeometry->isThere(GeomDetEnumerators::P2PXB) || trackerGeometry->isThere(GeomDetEnumerators::P2PXEC)) {
      phase_ = SiPixelPI::phase::two;
    } else if (trackerGeometry->isThere(GeomDetEnumerators::P1PXB) ||
               trackerGeometry->isThere(GeomDetEnumerators::P1PXEC)) {
      phase_ = SiPixelPI::phase::one;
    } else {
      phase_ = SiPixelPI::phase::zero;
    }

    // if it's a phase-2 geometry there are no phase-1 conditions
    if (phase_ == SiPixelPI::phase::two) {
      mode = 0;
    } else {
      if (doLatencyAnalysis_) {
        //SiStrip Latency
        const SiStripLatency *apvlat = &setup.getData(latencyToken_);
        if (apvlat->singleReadOutMode() == 1) {
          mode = 1;  // peak mode
        } else if (apvlat->singleReadOutMode() == 0) {
          mode = -1;  // deco mode
        }
      } else {
        mode = 0;
      }
    }

    conditionsMap_[run.run()].first = mode;
    conditionsMap_[run.run()].second = B_;

    // init the sipixel coordinates
    const TrackerTopology *trackerTopology = &setup.getData(trackerTopologyTokenBR_);
    const SiPixelFedCablingMap *siPixelFedCablingMap = &setup.getData(siPixelFedCablingMapTokenBR_);

    // Pixel Phase-1 helper class
    coord_.init(trackerTopology, trackerGeometry, siPixelFedCablingMap);
  }

  //*************************************************************
  void endRun(edm::Run const &, edm::EventSetup const &) override {}
  //*************************************************************

  //*************************************************************
  void beginJob() override
  //*************************************************************
  {
    if (DEBUG) {
      edm::LogInfo("GeneralPurposeTrackAnalyzer") << __LINE__ << std::endl;
    }

    TH1D::SetDefaultSumw2(kTRUE);
    etaMax_ = 3.0;

    ievt = 0;
    itrks = 0;

    hrun = book<TH1D>("h_run", "run", 100000, 230000, 240000);
    hlumi = book<TH1D>("h_lumi", "lumi", 1000, 0, 1000);

    hchi2ndof = book<TH1D>("h_chi2ndof", "chi2/ndf;#chi^{2}/ndf;tracks", 100, 0, 5.);
    hCharge = book<TH1D>("h_charge", "charge;Charge of the track;tracks", 5, -2.5, 2.5);
    hNtrk = book<TH1D>("h_Ntrk", "ntracks;Number of Tracks;events", 200, 0., 200.);
    hNtrkZoom = book<TH1D>("h_NtrkZoom", "Number of tracks; number of tracks;events", 10, 0., 10.);
    hNhighPurity =
        book<TH1D>("h_NhighPurity", "n. high purity tracks;Number of high purity tracks;events", 200, 0., 200.);

    htrkAlgo = book<TH1I>("h_trkAlgo",
                          "tracking step;iterative tracking step;tracks",
                          reco::TrackBase::algoSize,
                          0.,
                          double(reco::TrackBase::algoSize));

    htrkOriAlgo = book<TH1I>("h_trkOriAlgo",
                             "original tracking step;original iterative tracking step;tracks",
                             reco::TrackBase::algoSize,
                             0.,
                             double(reco::TrackBase::algoSize));

    for (size_t ibin = 0; ibin < reco::TrackBase::algoSize - 1; ibin++) {
      htrkAlgo->GetXaxis()->SetBinLabel(ibin + 1, (reco::TrackBase::algoNames[ibin]).c_str());
      htrkOriAlgo->GetXaxis()->SetBinLabel(ibin + 1, (reco::TrackBase::algoNames[ibin]).c_str());
    }

    htrkQuality = book<TH1I>("h_trkQuality", "track quality;track quality;tracks", 6, -1, 5);
    std::string qualities[7] = {"undef", "loose", "tight", "highPurity", "confirmed", "goodIterative"};
    for (int nbin = 1; nbin <= htrkQuality->GetNbinsX(); nbin++) {
      htrkQuality->GetXaxis()->SetBinLabel(nbin, (qualities[nbin - 1]).c_str());
    }

    hP = book<TH1D>("h_P", "Momentum;track momentum [GeV];tracks", 100, 0., 100.);
    hQoverP = book<TH1D>("h_qoverp", "Track q/p; track q/p [GeV^{-1}];tracks", 100, -1., 1.);
    hQoverPZoom = book<TH1D>("h_qoverpZoom", "Track q/p; track q/p [GeV^{-1}];tracks", 100, -0.1, 0.1);
    hPt = book<TH1D>("h_Pt", "Transverse Momentum;track p_{T} [GeV];tracks", 100, 0., 100.);
    hHit = book<TH1D>("h_nHits", "Number of hits;track n. hits;tracks", 50, -0.5, 49.5);
    hHit2D = book<TH1D>("h_nHit2D", "Number of 2D hits; number of 2D hits;tracks", 20, 0, 20);

    hHitCountVsZBPix = book<TH1D>("h_HitCountVsZBpix", "Number of BPix hits vs z;hit global z;hits", 60, -30, 30);
    hHitCountVsZFPix = book<TH1D>("h_HitCountVsZFpix", "Number of FPix hits vs z;hit global z;hits", 100, -100, 100);

    hHitCountVsXBPix = book<TH1D>("h_HitCountVsXBpix", "Number of BPix hits vs x;hit global x;hits", 20, -20, 20);
    hHitCountVsXFPix = book<TH1D>("h_HitCountVsXFpix", "Number of FPix hits vs x;hit global x;hits", 20, -20, 20);

    hHitCountVsYBPix = book<TH1D>("h_HitCountVsYBpix", "Number of BPix hits vs y;hit global y;hits", 20, -20, 20);
    hHitCountVsYFPix = book<TH1D>("h_HitCountVsYFpix", "Number of FPix hits vs y;hit global y;hits", 20, -20, 20);

    hHitCountVsThetaBPix =
        book<TH1D>("h_HitCountVsThetaBpix", "Number of BPix hits vs #theta;hit global #theta;hits", 20, 0., M_PI);
    hHitCountVsPhiBPix =
        book<TH1D>("h_HitCountVsPhiBpix", "Number of BPix hits vs #phi;hit global #phi;hits", 20, -M_PI, M_PI);

    hHitCountVsThetaFPix =
        book<TH1D>("h_HitCountVsThetaFpix", "Number of FPix hits vs #theta;hit global #theta;hits", 40, 0., M_PI);
    hHitCountVsPhiFPix =
        book<TH1D>("h_HitCountVsPhiFpix", "Number of FPix hits vs #phi;hit global #phi;hits", 20, -M_PI, M_PI);

    hEta = book<TH1D>("h_Eta", "Track pseudorapidity; track #eta;tracks", 100, -etaMax_, etaMax_);
    hPhi = book<TH1D>("h_Phi", "Track azimuth; track #phi;tracks", 100, -M_PI, M_PI);

    hPhiBarrel = book<TH1D>("h_PhiBarrel", "hPhiBarrel (0<|#eta|<0.8);track #Phi;tracks", 100, -M_PI, M_PI);
    hPhiOverlapPlus =
        book<TH1D>("h_PhiOverlapPlus", "hPhiOverlapPlus (0.8<#eta<1.4);track #phi;tracks", 100, -M_PI, M_PI);
    hPhiOverlapMinus =
        book<TH1D>("h_PhiOverlapMinus", "hPhiOverlapMinus (-1.4<#eta<-0.8);track #phi;tracks", 100, -M_PI, M_PI);
    hPhiEndcapPlus = book<TH1D>("h_PhiEndcapPlus", "hPhiEndcapPlus (#eta>1.4);track #phi;track", 100, -M_PI, M_PI);
    hPhiEndcapMinus = book<TH1D>("h_PhiEndcapMinus", "hPhiEndcapMinus (#eta<1.4);track #phi;tracks", 100, -M_PI, M_PI);

    if (!isCosmics_) {
      hPhp = book<TH1D>("h_P_hp", "Momentum (high purity);track momentum [GeV];tracks", 100, 0., 100.);
      hPthp = book<TH1D>("h_Pt_hp", "Transverse Momentum (high purity);track p_{T} [GeV];tracks", 100, 0., 100.);
      hHithp = book<TH1D>("h_nHit_hp", "Number of hits (high purity);track n. hits;tracks", 30, 0, 30);
      hEtahp = book<TH1D>("h_Eta_hp", "Track pseudorapidity (high purity); track #eta;tracks", 100, -etaMax_, etaMax_);
      hPhihp = book<TH1D>("h_Phi_hp", "Track azimuth (high purity); track #phi;tracks", 100, -M_PI, M_PI);
      hchi2ndofhp = book<TH1D>("h_chi2ndof_hp", "chi2/ndf (high purity);#chi^{2}/ndf;tracks", 100, 0, 5.);
      hchi2Probhp = book<TH1D>(
          "h_chi2_Prob_hp", "#chi^{2} probability (high purity);#chi^{2}prob_{Track};Number of Tracks", 100, 0.0, 1.);

      hvx = book<TH1D>("h_vx", "Track v_{x} ; track v_{x} [cm];tracks", 100, -1.5, 1.5);
      hvy = book<TH1D>("h_vy", "Track v_{y} ; track v_{y} [cm];tracks", 100, -1.5, 1.5);
      hvz = book<TH1D>("h_vz", "Track v_{z} ; track v_{z} [cm];tracks", 100, -20., 20.);
      hd0 = book<TH1D>("h_d0", "Track d_{0} ; track d_{0} [cm];tracks", 100, -1., 1.);
      hdxy = book<TH1D>("h_dxy", "Track d_{xy}; track d_{xy} [cm]; tracks", 100, -0.5, 0.5);
      hdz = book<TH1D>("h_dz", "Track d_{z} ; track d_{z} [cm]; tracks", 100, -20, 20);

      hd0PVvsphi =
          book<TH2D>("h2_d0PVvsphi", "hd0PVvsphi;track #phi;track d_{0}(PV) [cm]", 160, -M_PI, M_PI, 100, -1., 1.);
      hd0PVvseta =
          book<TH2D>("h2_d0PVvseta", "hdPV0vseta;track #eta;track d_{0}(PV) [cm]", 160, -2.5, 2.5, 100, -1., 1.);
      hd0PVvspt = book<TH2D>("h2_d0PVvspt", "hdPV0vspt;track p_{T};d_{0}(PV) [cm]", 50, 0., 100., 100, -1, 1.);

      hdxyBS = book<TH1D>("h_dxyBS", "hdxyBS; track d_{xy}(BS) [cm];tracks", 100, -0.1, 0.1);
      hd0BS = book<TH1D>("h_d0BS", "hd0BS ; track d_{0}(BS) [cm];tracks", 100, -0.1, 0.1);
      hdzBS = book<TH1D>("h_dzBS", "hdzBS ; track d_{z}(BS) [cm];tracks", 100, -12, 12);
      hdxyPV = book<TH1D>("h_dxyPV", "hdxyPV; track d_{xy}(PV) [cm];tracks", 100, -0.1, 0.1);
      hd0PV = book<TH1D>("h_d0PV", "hd0PV ; track d_{0}(PV) [cm];tracks", 100, -0.15, 0.15);
      hdzPV = book<TH1D>("h_dzPV", "hdzPV ; track d_{z}(PV) [cm];tracks", 100, -0.1, 0.1);

      hnhTIB = book<TH1D>("h_nHitTIB", "nhTIB;# hits in TIB; tracks", 20, 0., 20.);
      hnhTID = book<TH1D>("h_nHitTID", "nhTID;# hits in TID; tracks", 20, 0., 20.);
      hnhTOB = book<TH1D>("h_nHitTOB", "nhTOB;# hits in TOB; tracks", 20, 0., 20.);
      hnhTEC = book<TH1D>("h_nHitTEC", "nhTEC;# hits in TEC; tracks", 20, 0., 20.);

    } else {
      hvx = book<TH1D>("h_vx", "Track v_{x};track v_{x} [cm];tracks", 100, -100., 100.);
      hvy = book<TH1D>("h_vy", "Track v_{y};track v_{y} [cm];tracks", 100, -100., 100.);
      hvz = book<TH1D>("h_vz", "Track v_{z};track v_{z} [cm];track", 100, -100., 100.);
      hd0 = book<TH1D>("h_d0", "Track d_{0};track d_{0} [cm];track", 100, -100., 100.);
      hdxy = book<TH1D>("h_dxy", "Track d_{xy};track d_{xy} [cm];tracks", 100, -100, 100);
      hdz = book<TH1D>("h_dz", "Track d_{z};track d_{z} [cm];tracks", 100, -200, 200);

      hd0vsphi = book<TH2D>(
          "h2_d0vsphi", "Track d_{0} vs #phi; track #phi;track d_{0} [cm]", 160, -3.20, 3.20, 100, -100., 100.);
      hd0vseta = book<TH2D>(
          "h2_d0vseta", "Track d_{0} vs #eta; track #eta;track d_{0} [cm]", 160, -3.20, 3.20, 100, -100., 100.);
      hd0vspt =
          book<TH2D>("h2_d0vspt", "Track d_{0} vs p_{T};track p_{T};track d_{0} [cm]", 50, 0., 100., 100, -100, 100);

      hdxyBS = book<TH1D>("h_dxyBS", "Track d_{xy}(BS);d_{xy}(BS) [cm];tracks", 100, -100., 100.);
      hd0BS = book<TH1D>("h_d0BS", "Track d_{0}(BS);d_{0}(BS) [cm];tracks", 100, -100., 100.);
      hdzBS = book<TH1D>("h_dzBS", "Track d_{z}(BS);d_{z}(BS) [cm];tracks", 100, -100., 100.);
      hdxyPV = book<TH1D>("h_dxyPV", "Track d_{xy}(PV); d_{xy}(PV) [cm];tracks", 100, -100., 100.);
      hd0PV = book<TH1D>("h_d0PV", "Track d_{0}(PV); d_{0}(PV) [cm];tracks", 100, -100., 100.);
      hdzPV = book<TH1D>("h_dzPV", "Track d_{z}(PV); d_{z}(PV) [cm];tracks", 100, -100., 100.);

      hnhTIB = book<TH1D>("h_nHitTIB", "nhTIB;# hits in TIB; tracks", 30, 0., 30.);
      hnhTID = book<TH1D>("h_nHitTID", "nhTID;# hits in TID; tracks", 30, 0., 30.);
      hnhTOB = book<TH1D>("h_nHitTOB", "nhTOB;# hits in TOB; tracks", 30, 0., 30.);
      hnhTEC = book<TH1D>("h_nHitTEC", "nhTEC;# hits in TEC; tracks", 30, 0., 30.);
    }

    hnhpxb = book<TH1D>("h_nHitPXB", "nhpxb;# hits in Pixel Barrel; tracks", 10, 0., 10.);
    hnhpxe = book<TH1D>("h_nHitPXE", "nhpxe;# hits in Pixel Endcap; tracks", 10, 0., 10.);

    hHitComposition = book<TH1D>("h_hitcomposition", "track hit composition;;# hits", 6, -0.5, 5.5);

    pNBpixHitsVsVx =
        book<TProfile>("p_NpixHits_vs_Vx", "n. Barrel Pixel hits vs. v_{x};v_{x} (cm);n. BPix hits", 20, -20, 20);

    pNBpixHitsVsVy =
        book<TProfile>("p_NpixHits_vs_Vy", "n. Barrel Pixel hits vs. v_{y};v_{y} (cm);n. BPix hits", 20, -20, 20);

    pNBpixHitsVsVz =
        book<TProfile>("p_NpixHits_vs_Vz", "n. Barrel Pixel hits vs. v_{z};v_{z} (cm);n. BPix hits", 20, -100, 100);

    std::string dets[6] = {"PXB", "PXF", "TIB", "TID", "TOB", "TEC"};

    for (int i = 1; i <= hHitComposition->GetNbinsX(); i++) {
      hHitComposition->GetXaxis()->SetBinLabel(i, (dets[i - 1]).c_str());
    }

    // vector of track histograms

    // clang-format off

    vTrackHistos_.push_back(book<TH1F>("h_tracketa", "Track #eta;#eta_{Track};Number of Tracks", 90, -etaMax_, etaMax_));
    vTrackHistos_.push_back(book<TH1F>("h_trackphi", "Track #phi;#phi_{Track};Number of Tracks", 90, -M_PI, M_PI));
    vTrackHistos_.push_back(book<TH1F>("h_trackNumberOfValidHits", "Track # of valid hits;# of valid hits _{Track};Number of Tracks", 40, 0., 40.));
    vTrackHistos_.push_back(book<TH1F>("h_trackNumberOfLostHits", "Track # of lost hits;# of lost hits _{Track};Number of Tracks", 10, 0., 10.));
    vTrackHistos_.push_back(book<TH1F>("h_curvature", "Curvature #kappa;#kappa_{Track};Number of Tracks", 100, -.05, .05));
    vTrackHistos_.push_back(book<TH1F>("h_curvature_pos", "Curvature |#kappa| Positive Tracks;|#kappa_{pos Track}|;Number of Tracks", 100, .0, .05));
    vTrackHistos_.push_back(book<TH1F>("h_curvature_neg", "Curvature |#kappa| Negative Tracks;|#kappa_{neg Track}|;Number of Tracks", 100, .0, .05));
    vTrackHistos_.push_back(book<TH1F>("h_diff_curvature","Curvature |#kappa| Tracks Difference;|#kappa_{Track}|;# Pos Tracks - # Neg Tracks",100, .0, .05));
    vTrackHistos_.push_back(book<TH1F>("h_chi2", "Track #chi^{2};#chi^{2}_{Track};Number of Tracks", 500, -0.01, 500.));
    vTrackHistos_.push_back(book<TH1F>("h_chi2Prob", "#chi^{2} probability;Track Prob(#chi^{2},ndof);Number of Tracks", 100, 0.0, 1.));
    vTrackHistos_.push_back(book<TH1F>("h_normchi2", "#chi^{2}/ndof;#chi^{2}/ndof;Number of Tracks", 100, -0.01, 10.));
    //variable binning for chi2/ndof vs. pT
    double xBins[19] = {0., 0.15, 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 7., 10., 15., 25., 40., 100., 200.};
    vTrackHistos_.push_back(book<TH1F>("h_pt", "Track p_{T};p_{T}^{track} [GeV];Number of Tracks", 250, 0., 250));
    vTrackHistos_.push_back(book<TH1F>("h_ptrebin", "Track p_{T};p_{T}^{track} [GeV];Number of Tracks", 18, xBins));
    vTrackHistos_.push_back(book<TH1F>("h_ptResolution", "#delta_{p_{T}}/p_{T}^{track};#delta_{p_{T}}/p_{T}^{track};Number of Tracks", 100, 0., 0.5));

    vTrackProfiles_.push_back(book<TProfile>("p_d0_vs_phi", "Transverse Impact Parameter vs. #phi;#phi_{Track};#LT d_{0} #GT [cm]", 100, -M_PI, M_PI));
    vTrackProfiles_.push_back(book<TProfile>("p_dz_vs_phi", "Longitudinal Impact Parameter vs. #phi;#phi_{Track};#LT d_{z} #GT [cm]", 100, -M_PI, M_PI));
    vTrackProfiles_.push_back(book<TProfile>("p_d0_vs_eta", "Transverse Impact Parameter vs. #eta;#eta_{Track};#LT d_{0} #GT [cm]", 100, -etaMax_, etaMax_));
    vTrackProfiles_.push_back(book<TProfile>("p_dz_vs_eta", "Longitudinal Impact Parameter vs. #eta;#eta_{Track};#LT d_{z} #GT [cm]", 100, -etaMax_, etaMax_));
    vTrackProfiles_.push_back(book<TProfile>("p_chi2_vs_phi", "#chi^{2} vs. #phi;#phi_{Track};#LT #chi^{2} #GT", 100, -M_PI, M_PI));
    vTrackProfiles_.push_back(book<TProfile>("p_chi2Prob_vs_phi", "#chi^{2} probablility vs. #phi;#phi_{Track};#LT #chi^{2} probability#GT", 100, -M_PI, M_PI));
    vTrackProfiles_.push_back(book<TProfile>("p_chi2Prob_vs_d0", "#chi^{2} probablility vs. |d_{0}|;|d_{0}|[cm];#LT #chi^{2} probability#GT", 100, 0, 80));
    vTrackProfiles_.push_back(book<TProfile>("p_chi2Prob_vs_dz", "#chi^{2} probablility vs. dz;d_{z} [cm];#LT #chi^{2} probability#GT", 100, -30, 30));
    vTrackProfiles_.push_back(book<TProfile>("p_normchi2_vs_phi", "#chi^{2}/ndof vs. #phi;#phi_{Track};#LT #chi^{2}/ndof #GT", 100, -M_PI, M_PI));
    vTrackProfiles_.push_back(book<TProfile>("p_chi2_vs_eta", "#chi^{2} vs. #eta;#eta_{Track};#LT #chi^{2} #GT", 100, -etaMax_, etaMax_));
    vTrackProfiles_.push_back(book<TProfile>("p_normchi2_vs_pt", "norm #chi^{2} vs. p_{T}_{Track}; p_{T}_{Track};#LT #chi^{2}/ndof #GT", 18, xBins));
    vTrackProfiles_.push_back(book<TProfile>("p_normchi2_vs_p", "#chi^{2}/ndof vs. p_{Track};p_{Track};#LT #chi^{2}/ndof #GT", 18, xBins));
    vTrackProfiles_.push_back(book<TProfile>("p_chi2Prob_vs_eta","#chi^{2} probability vs. #eta;#eta_{Track};#LT #chi^{2} probability #GT",100,-etaMax_,etaMax_));
    vTrackProfiles_.push_back(book<TProfile>("p_normchi2_vs_eta", "#chi^{2}/ndof vs. #eta;#eta_{Track};#LT #chi^{2}/ndof #GT", 100, -etaMax_, etaMax_));
    vTrackProfiles_.push_back(book<TProfile>("p_kappa_vs_phi", "#kappa vs. #phi;#phi_{Track};#kappa", 100, -M_PI, M_PI));
    vTrackProfiles_.push_back(book<TProfile>("p_kappa_vs_eta", "#kappa vs. #eta;#eta_{Track};#kappa", 100, -etaMax_, etaMax_));
    vTrackProfiles_.push_back(book<TProfile>("p_ptResolution_vs_phi","#delta_{p_{T}}/p_{T}^{track};#phi^{track};#delta_{p_{T}}/p_{T}^{track}",100,-M_PI,M_PI));
    vTrackProfiles_.push_back(book<TProfile>("p_ptResolution_vs_eta","#delta_{p_{T}}/p_{T}^{track};#eta^{track};#delta_{p_{T}}/p_{T}^{track}",100,-etaMax_,etaMax_));

    vTrack2DHistos_.push_back(book<TH2F>("h2_phi_vs_eta", "Track #phi vs. #eta;#eta_{Track};#phi_{Track}",50, -etaMax_, etaMax_, 50, -M_PI, M_PI));
    vTrack2DHistos_.push_back(book<TH2F>("h2_d0_vs_phi", "Transverse Impact Parameter vs. #phi;#phi_{Track};d_{0} [cm]", 100, -M_PI, M_PI, 100, -1., 1.));
    vTrack2DHistos_.push_back(book<TH2F>("h2_dz_vs_phi", "Longitudinal Impact Parameter vs. #phi;#phi_{Track};d_{z} [cm]",100, -M_PI, M_PI, 100, -100., 100.));
    vTrack2DHistos_.push_back(book<TH2F>("h2_d0_vs_eta", "Transverse Impact Parameter vs. #eta;#eta_{Track};d_{0} [cm]", 100, -etaMax_, etaMax_, 100, -1., 1.));
    vTrack2DHistos_.push_back(book<TH2F>("h2_dz_vs_eta", "Longitudinal Impact Parameter vs. #eta;#eta_{Track};d_{z} [cm]", 100, -etaMax_, etaMax_, 100, -100., 100.));
    vTrack2DHistos_.push_back(book<TH2F>("h2_chi2_vs_phi", "#chi^{2} vs. #phi;#phi_{Track};#chi^{2}", 100, -M_PI, M_PI, 500, 0., 500.));
    vTrack2DHistos_.push_back(book<TH2F>("h2_chi2Prob_vs_phi", "#chi^{2} probability vs. #phi;#phi_{Track};#chi^{2} probability", 100, -M_PI, M_PI, 100, 0., 1.));
    vTrack2DHistos_.push_back(book<TH2F>("h2_chi2Prob_vs_d0", "#chi^{2} probability vs. |d_{0}|;|d_{0}| [cm];#chi^{2} probability", 100, 0, 80, 100, 0., 1.));
    vTrack2DHistos_.push_back(book<TH2F>("h2_normchi2_vs_phi", "#chi^{2}/ndof vs. #phi;#phi_{Track};#chi^{2}/ndof", 100, -M_PI, M_PI, 100, 0., 10.));
    vTrack2DHistos_.push_back(book<TH2F>("h2_chi2_vs_eta", "#chi^{2} vs. #eta;#eta_{Track};#chi^{2}", 100, -etaMax_, etaMax_, 500, 0., 500.));
    vTrack2DHistos_.push_back(book<TH2F>("h2_chi2Prob_vs_eta", "#chi^{2} probaility vs. #eta;#eta_{Track};#chi^{2} probability", 100, -etaMax_, etaMax_, 100, 0., 1.));
    vTrack2DHistos_.push_back(book<TH2F>("h2_normchi2_vs_eta", "#chi^{2}/ndof vs. #eta;#eta_{Track};#chi^{2}/ndof", 100, -etaMax_, etaMax_, 100, 0., 10.));
    vTrack2DHistos_.push_back(book<TH2F>("h2_kappa_vs_phi", "#kappa vs. #phi;#phi_{Track};#kappa", 100, -M_PI, M_PI, 100, .0, .05));
    vTrack2DHistos_.push_back(book<TH2F>("h2_kappa_vs_eta", "#kappa vs. #eta;#eta_{Track};#kappa", 100, -etaMax_, etaMax_, 100, .0, .05));
    vTrack2DHistos_.push_back(book<TH2F>("h2_normchi2_vs_kappa", "#kappa vs. #chi^{2}/ndof;#chi^{2}/ndof;#kappa", 100, 0., 10, 100, -.03, .03));

    // clang-format on

  }  //beginJob

  //*************************************************************
  void endJob() override
  //*************************************************************
  {
    edm::LogPrint("GeneralPurposeTrackAnalyzer") << "*******************************" << std::endl;
    edm::LogPrint("GeneralPurposeTrackAnalyzer") << "Events run in total: " << ievt << std::endl;
    edm::LogPrint("GeneralPurposeTrackAnalyzer") << "n. tracks: " << itrks << std::endl;
    edm::LogPrint("GeneralPurposeTrackAnalyzer") << "*******************************" << std::endl;

    int nFiringTriggers = triggerMap_.size();
    edm::LogPrint("GeneralPurposeTrackAnalyzer") << "firing triggers: " << nFiringTriggers << std::endl;
    edm::LogPrint("GeneralPurposeTrackAnalyzer") << "*******************************" << std::endl;

    tksByTrigger_ =
        book<TH1D>("tksByTrigger", "tracks by HLT path;;% of # traks", nFiringTriggers, -0.5, nFiringTriggers - 0.5);
    evtsByTrigger_ =
        book<TH1D>("evtsByTrigger", "events by HLT path;;% of # events", nFiringTriggers, -0.5, nFiringTriggers - 0.5);

    int i = 0;
    for (const auto &it : triggerMap_) {
      i++;

      double trkpercent = ((it.second).second) * 100. / double(itrks);
      double evtpercent = ((it.second).first) * 100. / double(ievt);

      std::cout.precision(4);

      edm::LogPrint("GeneralPurposeTrackAnalyzer")
          << "HLT path: " << std::setw(60) << std::left << it.first << " | events firing: " << std::right
          << std::setw(8) << (it.second).first << " (" << std::setw(8) << std::fixed << std::setprecision(4)
          << evtpercent << "%)"
          << " | tracks collected: " << std::setw(10) << (it.second).second << " (" << std::setw(8) << std::fixed
          << std::setprecision(4) << trkpercent << "%)";

      tksByTrigger_->SetBinContent(i, trkpercent);
      tksByTrigger_->GetXaxis()->SetBinLabel(i, (it.first).c_str());

      evtsByTrigger_->SetBinContent(i, evtpercent);
      evtsByTrigger_->GetXaxis()->SetBinLabel(i, (it.first).c_str());
    }

    int nRuns = conditionsMap_.size();

    std::vector<int> theRuns_;
    for (const auto &it : conditionsMap_) {
      theRuns_.push_back(it.first);
    }

    sort(theRuns_.begin(), theRuns_.end());
    int runRange = theRuns_.back() - theRuns_.front() + 1;

    edm::LogPrint("GeneralPurposeTrackAnalyzer") << "*******************************" << std::endl;
    edm::LogPrint("GeneralPurposeTrackAnalyzer") << "first run: " << theRuns_.front() << std::endl;
    edm::LogPrint("GeneralPurposeTrackAnalyzer") << "last run:  " << theRuns_.back() << std::endl;
    edm::LogPrint("GeneralPurposeTrackAnalyzer") << "considered runs: " << nRuns << std::endl;
    edm::LogPrint("GeneralPurposeTrackAnalyzer") << "*******************************" << std::endl;

    modeByRun_ = book<TH1D>("modeByRun",
                            "Strip APV mode by run number;;APV mode (-1=deco,+1=peak)",
                            runRange,
                            theRuns_.front() - 0.5,
                            theRuns_.back() + 0.5);

    fieldByRun_ = book<TH1D>("fieldByRun",
                             "CMS B-field intensity by run number;;B-field intensity [T]",
                             runRange,
                             theRuns_.front() - 0.5,
                             theRuns_.back() + 0.5);

    for (const auto &the_r : theRuns_) {
      if (conditionsMap_.find(the_r)->second.first != 0) {
        edm::LogPrint("GeneralPurposeTrackAnalyzer")
            << "run:" << the_r << " | isPeak: " << std::setw(4) << conditionsMap_.find(the_r)->second.first
            << "| B-field: " << conditionsMap_.find(the_r)->second.second << " [T]"
            << "| events: " << std::setw(10) << runInfoMap_.find(the_r)->second.first << ", tracks " << std::setw(10)
            << runInfoMap_.find(the_r)->second.second << std::endl;
      }

      modeByRun_->SetBinContent((the_r - theRuns_.front()) + 1, conditionsMap_.find(the_r)->second.first);
      fieldByRun_->SetBinContent((the_r - theRuns_.front()) + 1, conditionsMap_.find(the_r)->second.second);
      modeByRun_->GetXaxis()->SetBinLabel((the_r - theRuns_.front()) + 1, std::to_string(the_r).c_str());
      fieldByRun_->GetXaxis()->SetBinLabel((the_r - theRuns_.front()) + 1, std::to_string(the_r).c_str());
    }

    if (phase_ < SiPixelPI::phase::two) {
      if (phase_ == SiPixelPI::phase::zero) {
        pmap->save(true, 0, 0, "PixelHitMap.pdf", 600, 800);
        pmap->save(true, 0, 0, "PixelHitMap.png", 500, 750);
      }

      tmap->save(true, 0, 0, "StripHitMap.pdf");
      tmap->save(true, 0, 0, "StripHitMap.png");

      gStyle->SetPalette(kRainBow);
      pixelmap->beautifyAllHistograms();

      TCanvas cB("CanvBarrel", "CanvBarrel", 1200, 1000);
      pixelmap->drawBarrelMaps("entriesBarrel", cB);
      cB.SaveAs("pixelBarrelEntries.png");

      TCanvas cF("CanvForward", "CanvForward", 1600, 1000);
      pixelmap->drawForwardMaps("entriesForward", cF);
      cF.SaveAs("pixelForwardEntries.png");

      TCanvas cRocs = TCanvas("cRocs", "cRocs", 1200, 1600);
      pixelrocsmap_->drawMaps(cRocs, "Pixel on-track clusters occupancy");
      cRocs.SaveAs("Phase1PixelROCMaps_fullROCs.png");
    }
  }

  //*************************************************************
  bool isHit2D(const TrackingRecHit &hit)
  //*************************************************************
  {
    bool countStereoHitAs2D_ = true;
    // we count SiStrip stereo modules as 2D if selected via countStereoHitAs2D_
    // (since they provide theta information)
    if (!hit.isValid() ||
        (hit.dimension() < 2 && !countStereoHitAs2D_ && !dynamic_cast<const SiStripRecHit1D *>(&hit))) {
      return false;  // real RecHit1D - but SiStripRecHit1D depends on countStereoHitAs2D_
    } else {
      const DetId detId(hit.geographicalId());
      if (detId.det() == DetId::Tracker) {
        if (detId.subdetId() == kBPIX || detId.subdetId() == kFPIX) {
          return true;  // pixel is always 2D
        } else {        // should be SiStrip now
          const SiStripDetId stripId(detId);
          if (stripId.stereo())
            return countStereoHitAs2D_;  // stereo modules
          else if (dynamic_cast<const SiStripRecHit1D *>(&hit) || dynamic_cast<const SiStripRecHit2D *>(&hit))
            return false;  // rphi modules hit
          //the following two are not used any more since ages...
          else if (dynamic_cast<const SiStripMatchedRecHit2D *>(&hit))
            return true;  // matched is 2D
          else if (dynamic_cast<const ProjectedSiStripRecHit2D *>(&hit)) {
            const ProjectedSiStripRecHit2D *pH = static_cast<const ProjectedSiStripRecHit2D *>(&hit);
            return (countStereoHitAs2D_ && this->isHit2D(pH->originalHit()));  // depends on original...
          } else {
            edm::LogError("UnknownType") << "@SUB=GeneralPurposeTrackAnalyzer::isHit2D"
                                         << "Tracker hit not in pixel, neither SiStripRecHit[12]D nor "
                                         << "SiStripMatchedRecHit2D nor ProjectedSiStripRecHit2D.";
            return false;
          }
        }
      } else {  // not tracker??
        edm::LogWarning("DetectorMismatch") << "@SUB=GeneralPurposeTrackAnalyzer::isHit2D"
                                            << "Hit not in tracker with 'official' dimension >=2.";
        return true;  // dimension() >= 2 so accept that...
      }
    }
    // never reached...
  }
};

//*************************************************************
void GeneralPurposeTrackAnalyzer::fillDescriptions(edm::ConfigurationDescriptions &descriptions)
//*************************************************************
{
  edm::ParameterSetDescription desc;
  desc.setComment("Generic track analyzer to check ALCARECO sample quantities");
  desc.add<edm::InputTag>("TkTag", edm::InputTag("generalTracks"));
  desc.add<edm::InputTag>("TriggerResultsTag", edm::InputTag("TriggerResults", "", "HLT"));
  desc.add<edm::InputTag>("BeamSpotTag", edm::InputTag("offlineBeamSpot"));
  desc.add<edm::InputTag>("VerticesTag", edm::InputTag("offlinePrimaryVertices"));
  desc.add<bool>("isCosmics", false);
  desc.add<bool>("doLatencyAnalysis", true);
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(GeneralPurposeTrackAnalyzer);

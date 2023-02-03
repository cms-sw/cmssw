// -*- C++ -*-
//
// Package:    Alignment/OfflineValidation
// Class:      DMRChecker
//
/**\class DMRChecker DMRChecker.cc Alignment/OfflineValidation/plugins/DMRChecker.cc

*/
//
// Original Author:  Marco Musich
//         Created:  Mon, 10 Aug 2020 15:45:00 GMT
//
//

// ROOT includes

#include "RtypesCore.h"
#include "TAxis.h"
#include "TCanvas.h"
#include "TColor.h"
#include "TH1.h"
#include "TH2.h"
#include "TLatex.h"
#include "TMath.h"
#include "TProfile.h"
#include "TString.h"
#include "TStyle.h"
#include "TVirtualPad.h"

// STL includes

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <ext/alloc_traits.h>
#include <fmt/printf.h>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include "boost/range/adaptor/indexed.hpp"

// user system includes

#include "CalibTracker/StandaloneTrackerTopology/interface/StandaloneTrackerTopology.h"
#include "CommonTools/TrackerMap/interface/TrackerMap.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CommonTools/Utils/interface/TFileDirectory.h"
#include "DQM/TrackerRemapper/interface/Phase1PixelMaps.h"
#include "DQM/TrackerRemapper/interface/Phase1PixelSummaryMap.h"
#include "CondCore/SiPixelPlugins/interface/PixelRegionContainers.h"
#include "CondCore/SiPixelPlugins/interface/SiPixelPayloadInspectorHelper.h"
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "CondFormats/SiStripObjects/interface/SiStripLatency.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefTraits.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/Phi.h"
#include "DataFormats/GeometryVector/interface/Theta.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Provenance/interface/RunLumiEventNumber.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackResiduals.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/ErrorObj.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFiller.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/CommonTopologies/interface/GeomDet.h"
#include "Geometry/CommonTopologies/interface/TrackerGeomDet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#define DEBUG 0

using namespace std;
using namespace edm;

const int kBPIX = PixelSubdetector::PixelBarrel;
const int kFPIX = PixelSubdetector::PixelEndcap;
constexpr float cmToUm = 10000.;

/**
 * Auxilliary POD to store the data for
 * the running mean algorithm.
 */

namespace running {
  struct Estimators {
    int rDirection;
    int zDirection;
    int rOrZDirection;
    int hitCount;
    float runningMeanOfRes_;
    float runningVarOfRes_;
    float runningNormMeanOfRes_;
    float runningNormVarOfRes_;
  };

  using estimatorMap = std::map<uint32_t, running::Estimators>;

}  // namespace running

class DMRChecker : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
public:
  DMRChecker(const edm::ParameterSet &pset)
      : geomToken_(esConsumes<edm::Transition::BeginRun>()),
        runInfoToken_(esConsumes<edm::Transition::BeginRun>()),
        magFieldToken_(esConsumes<edm::Transition::BeginRun>()),
        topoToken_(esConsumes<edm::Transition::BeginRun>()),
        latencyToken_(esConsumes<edm::Transition::BeginRun>()),
        isCosmics_(pset.getParameter<bool>("isCosmics")) {
    usesResource(TFileService::kSharedResource);

    TkTag_ = pset.getParameter<edm::InputTag>("TkTag");
    theTrackCollectionToken_ = consumes<reco::TrackCollection>(TkTag_);

    TriggerResultsTag_ = pset.getParameter<edm::InputTag>("TriggerResultsTag");
    hltresultsToken_ = consumes<edm::TriggerResults>(TriggerResultsTag_);

    BeamSpotTag_ = pset.getParameter<edm::InputTag>("BeamSpotTag");
    beamspotToken_ = consumes<reco::BeamSpot>(BeamSpotTag_);

    VerticesTag_ = pset.getParameter<edm::InputTag>("VerticesTag");
    vertexToken_ = consumes<reco::VertexCollection>(VerticesTag_);

    // initialize conventional Tracker maps

    pmap = std::make_unique<TrackerMap>("Pixel");
    pmap->onlyPixel(true);
    pmap->setTitle("Pixel Hit entries");
    pmap->setPalette(1);

    tmap = std::make_unique<TrackerMap>("Strip");
    tmap->setTitle("Strip Hit entries");
    tmap->setPalette(1);

    // initialize Phase1 Pixel Maps

    pixelmap = std::make_unique<Phase1PixelMaps>("COLZ0 L");
    pixelmap->bookBarrelHistograms("DMRsX", "Median Residuals x-direction", "Median Residuals");
    pixelmap->bookForwardHistograms("DMRsX", "Median Residuals x-direction", "Median Residuals");

    pixelmap->bookBarrelHistograms("DMRsY", "Median Residuals y-direction", "Median Residuals");
    pixelmap->bookForwardHistograms("DMRsY", "Median Residuals y-direction", "Median Residuals");

    // set no rescale
    pixelmap->setNoRescale();

    // initialize Full Pixel Map
    fullPixelmapXDMR = std::make_unique<Phase1PixelSummaryMap>("", "DMR-x", "median of residuals [#mum]");
    fullPixelmapYDMR = std::make_unique<Phase1PixelSummaryMap>("", "DMR-y", "median of residuals [#mum]");
  }

  static void fillDescriptions(edm::ConfigurationDescriptions &);

  ~DMRChecker() override = default;

  /*_______________________________________________________
  //
  // auxilliary method to retrieve certain histogram types
  //_______________________________________________________
  */
  template <class OBJECT_TYPE>
  int index(const std::vector<OBJECT_TYPE *> &vec, const std::string &name) {
    for (const auto &iter : vec | boost::adaptors::indexed(0)) {
      if (iter.value() && iter.value()->GetName() == name) {
        return iter.index();
      }
    }
    edm::LogError("Alignment") << "@SUB=DMRChecker::index"
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
  const edm::ESGetToken<RunInfo, RunInfoRcd> runInfoToken_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magFieldToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;
  const edm::ESGetToken<SiStripLatency, SiStripLatencyRcd> latencyToken_;

  const MagneticField *magneticField_;
  const TrackerGeometry *trackerGeometry_;
  const TrackerTopology *trackerTopology_;

  edm::Service<TFileService> fs;

  std::unique_ptr<Phase1PixelMaps> pixelmap;
  std::unique_ptr<Phase1PixelSummaryMap> fullPixelmapXDMR;
  std::unique_ptr<Phase1PixelSummaryMap> fullPixelmapYDMR;

  std::unique_ptr<PixelRegions::PixelRegionContainers> PixelDMRS_x_ByLayer;
  std::unique_ptr<PixelRegions::PixelRegionContainers> PixelDMRS_y_ByLayer;

  std::unique_ptr<TrackerMap> tmap;
  std::unique_ptr<TrackerMap> pmap;

  TH1D *hchi2ndof;
  TH1D *hNtrk;
  TH1D *hNtrkZoom;
  TH1I *htrkQuality;
  TH1I *htrkAlgo;
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

  TH1D *hBPixResXPrime;
  TH1D *hFPixResXPrime;
  TH1D *hFPixZPlusResXPrime;
  TH1D *hFPixZMinusResXPrime;

  TH1D *hBPixResYPrime;
  TH1D *hFPixResYPrime;
  TH1D *hFPixZPlusResYPrime;
  TH1D *hFPixZMinusResYPrime;

  TH1D *hBPixResXPull;
  TH1D *hFPixResXPull;
  TH1D *hFPixZPlusResXPull;
  TH1D *hFPixZMinusResXPull;

  TH1D *hBPixResYPull;
  TH1D *hFPixResYPull;
  TH1D *hFPixZPlusResYPull;
  TH1D *hFPixZMinusResYPull;

  TH1D *hTIBResXPrime;
  TH1D *hTOBResXPrime;
  TH1D *hTIDResXPrime;
  TH1D *hTECResXPrime;

  TH1D *hTIBResXPull;
  TH1D *hTOBResXPull;
  TH1D *hTIDResXPull;
  TH1D *hTECResXPull;

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

  TH1D *hHitCountVsXFPixPlus;
  TH1D *hHitCountVsXFPixMinus;
  TH1D *hHitCountVsYFPixPlus;
  TH1D *hHitCountVsYFPixMinus;
  TH1D *hHitCountVsZFPixPlus;
  TH1D *hHitCountVsZFPixMinus;

  TH1D *hHitCountVsThetaFPixPlus;
  TH1D *hHitCountVsPhiFPixPlus;

  TH1D *hHitCountVsThetaFPixMinus;
  TH1D *hHitCountVsPhiFPixMinus;

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

  TH1D *tracksByRun_;
  TH1D *hitsByRun_;

  TH1D *trackRatesByRun_;
  TH1D *eventRatesByRun_;

  TH1D *hitsinBPixByRun_;
  TH1D *hitsinFPixByRun_;

  // Pixel

  TH1D *DMRBPixX_;
  TH1D *DMRBPixY_;

  TH1D *DMRFPixX_;
  TH1D *DMRFPixY_;

  TH1D *DRnRBPixX_;
  TH1D *DRnRBPixY_;

  TH1D *DRnRFPixX_;
  TH1D *DRnRFPixY_;

  // Strips

  TH1D *DMRTIB_;
  TH1D *DMRTOB_;

  TH1D *DMRTID_;
  TH1D *DMRTEC_;

  TH1D *DRnRTIB_;
  TH1D *DRnRTOB_;

  TH1D *DRnRTID_;
  TH1D *DRnRTEC_;

  // Split DMRs

  std::array<TH1D *, 2> DMRBPixXSplit_;
  std::array<TH1D *, 2> DMRBPixYSplit_;

  std::array<TH1D *, 2> DMRFPixXSplit_;
  std::array<TH1D *, 2> DMRFPixYSplit_;

  std::array<TH1D *, 2> DMRTIBSplit_;
  std::array<TH1D *, 2> DMRTOBSplit_;

  // residuals

  std::map<unsigned int, TH1D *> barrelLayersResidualsX;
  std::map<unsigned int, TH1D *> barrelLayersPullsX;
  std::map<unsigned int, TH1D *> barrelLayersResidualsY;
  std::map<unsigned int, TH1D *> barrelLayersPullsY;

  std::map<unsigned int, TH1D *> endcapDisksResidualsX;
  std::map<unsigned int, TH1D *> endcapDisksPullsX;
  std::map<unsigned int, TH1D *> endcapDisksResidualsY;
  std::map<unsigned int, TH1D *> endcapDisksPullsY;

  int ievt;
  int itrks;
  int mode;
  bool firstEvent_;

  SiPixelPI::phase phase_;
  float etaMax_;

  const bool isCosmics_;

  edm::InputTag TkTag_;
  edm::InputTag TriggerResultsTag_;
  edm::InputTag BeamSpotTag_;
  edm::InputTag VerticesTag_;

  edm::EDGetTokenT<reco::TrackCollection> theTrackCollectionToken_;
  edm::EDGetTokenT<edm::TriggerResults> hltresultsToken_;
  edm::EDGetTokenT<reco::BeamSpot> beamspotToken_;
  edm::EDGetTokenT<reco::VertexCollection> vertexToken_;

  std::map<std::string, std::pair<int, int> > triggerMap_;
  std::map<int, std::pair<int, float> > conditionsMap_;
  std::map<int, std::pair<int, int> > runInfoMap_;
  std::map<int, std::array<int, 6> > runHitsMap_;

  std::map<int, float> timeMap_;

  // Pixel

  running::estimatorMap resDetailsBPixX_;
  running::estimatorMap resDetailsBPixY_;
  running::estimatorMap resDetailsFPixX_;
  running::estimatorMap resDetailsFPixY_;

  // Strips

  running::estimatorMap resDetailsTIB_;
  running::estimatorMap resDetailsTOB_;
  running::estimatorMap resDetailsTID_;
  running::estimatorMap resDetailsTEC_;

  void analyze(const edm::Event &event, const edm::EventSetup &setup) override {
    ievt++;

    edm::Handle<reco::TrackCollection> trackCollection = event.getHandle(theTrackCollectionToken_);

    if (firstEvent_) {
      if (trackerGeometry_->isThere(GeomDetEnumerators::P2PXB) ||
          trackerGeometry_->isThere(GeomDetEnumerators::P2PXEC)) {
        phase_ = SiPixelPI::phase::two;
      } else if (trackerGeometry_->isThere(GeomDetEnumerators::P1PXB) ||
                 trackerGeometry_->isThere(GeomDetEnumerators::P1PXEC)) {
        phase_ = SiPixelPI::phase::one;
      } else {
        phase_ = SiPixelPI::phase::zero;
      }
      firstEvent_ = false;
    }

    GlobalPoint zeroPoint(0, 0, 0);
    if (DEBUG)
      edm::LogVerbatim("DMRChecker") << "event #" << ievt << " Event ID = " << event.id()
                                     << " magnetic field: " << magneticField_->inTesla(zeroPoint) << std::endl;

    const reco::TrackCollection tC = *(trackCollection.product());
    itrks += tC.size();

    runInfoMap_[event.run()].first += 1;
    runInfoMap_[event.run()].second += tC.size();

    if (DEBUG)
      edm::LogVerbatim("DMRChecker") << "Reconstructed " << tC.size() << " tracks" << std::endl;

    edm::Handle<edm::TriggerResults> hltresults = event.getHandle(hltresultsToken_);
    if (hltresults.isValid()) {
      const edm::TriggerNames &triggerNames_ = event.triggerNames(*hltresults);
      int ntrigs = hltresults->size();

      for (int itrig = 0; itrig != ntrigs; ++itrig) {
        const string &trigName = triggerNames_.triggerName(itrig);
        bool accept = hltresults->accept(itrig);
        if (accept == 1) {
          if (DEBUG)
            edm::LogVerbatim("DMRChecker") << trigName << " " << accept << " ,track size: " << tC.size() << endl;
          triggerMap_[trigName].first += 1;
          triggerMap_[trigName].second += tC.size();
        }
      }
    }

    hrun->Fill(event.run());
    hlumi->Fill(event.luminosityBlock());

    int nHighPurityTracks = 0;

    for (const auto &track : tC) {
      auto const &residuals = track.extra()->residuals();

      unsigned int nHit2D = 0;
      int h_index = 0;
      for (trackingRecHit_iterator iHit = track.recHitsBegin(); iHit != track.recHitsEnd(); ++iHit, ++h_index) {
        if (this->isHit2D(**iHit))
          ++nHit2D;

        double resX = residuals.residualX(h_index);
        double resY = residuals.residualY(h_index);
        double pullX = residuals.pullX(h_index);
        double pullY = residuals.pullY(h_index);

        const DetId &detId = (*iHit)->geographicalId();

        unsigned int subid = detId.subdetId();
        uint32_t detid_db = detId.rawId();

        const GeomDet *geomDet(trackerGeometry_->idToDet(detId));

        float uOrientation(-999.F), vOrientation(-999.F);
        LocalPoint lPModule(0., 0., 0.), lUDirection(1., 0., 0.), lVDirection(0., 1., 0.), lWDirection(0., 0., 1.);

        // do all the transformations here
        GlobalPoint gUDirection = geomDet->surface().toGlobal(lUDirection);
        GlobalPoint gVDirection = geomDet->surface().toGlobal(lVDirection);
        GlobalPoint gWDirection = geomDet->surface().toGlobal(lWDirection);
        GlobalPoint gPModule = geomDet->surface().toGlobal(lPModule);

        if (!(*iHit)->detUnit())
          continue;  // is it a single physical module?

        if ((*iHit)->isValid() && (subid > PixelSubdetector::PixelEndcap)) {
          tmap->fill(detid_db, 1);

          //LocalPoint lp = (*iHit)->localPosition();
          //LocalError le = (*iHit)->localPositionError();

          // fill DMRs and DrNRs
          if (subid == StripSubdetector::TIB) {
            uOrientation = deltaPhi(gUDirection.barePhi(), gPModule.barePhi()) >= 0. ? +1.F : -1.F;
            //vOrientation = gVDirection.z() - gPModule.z() >= 0 ? +1.F : -1.F; // not used for Strips

            // if the detid has never occcurred yet, set the local orientations
            if (resDetailsTIB_.find(detid_db) == resDetailsTIB_.end()) {
              resDetailsTIB_[detid_db].rDirection = gWDirection.perp() - gPModule.perp() >= 0 ? +1 : -1;
              resDetailsTIB_[detid_db].zDirection = gVDirection.z() - gPModule.z() >= 0 ? +1 : -1;
              resDetailsTIB_[detid_db].rOrZDirection = resDetailsTIB_[detid_db].rDirection;  // barrel (split in r)
            }

            hTIBResXPrime->Fill(uOrientation * resX * cmToUm);
            hTIBResXPull->Fill(pullX);

            // update residuals
            this->updateOnlineMomenta(resDetailsTIB_, detid_db, uOrientation * resX * cmToUm, pullX);

          } else if (subid == StripSubdetector::TOB) {
            uOrientation = deltaPhi(gUDirection.barePhi(), gPModule.barePhi()) >= 0. ? +1.F : -1.F;
            //vOrientation = gVDirection.z() - gPModule.z() >= 0 ? +1.F : -1.F; // not used for Strips

            hTOBResXPrime->Fill(uOrientation * resX * cmToUm);
            hTOBResXPull->Fill(pullX);

            // if the detid has never occcurred yet, set the local orientations
            if (resDetailsTOB_.find(detid_db) == resDetailsTOB_.end()) {
              resDetailsTOB_[detid_db].rDirection = gWDirection.perp() - gPModule.perp() >= 0 ? +1 : -1;
              resDetailsTOB_[detid_db].zDirection = gVDirection.z() - gPModule.z() >= 0 ? +1 : -1;
              resDetailsTOB_[detid_db].rOrZDirection = resDetailsTOB_[detid_db].rDirection;  // barrel (split in r)
            }

            // update residuals
            this->updateOnlineMomenta(resDetailsTOB_, detid_db, uOrientation * resX * cmToUm, pullX);

          } else if (subid == StripSubdetector::TID) {
            uOrientation = deltaPhi(gUDirection.barePhi(), gPModule.barePhi()) >= 0. ? +1.F : -1.F;
            //vOrientation = gVDirection.perp() - gPModule.perp() >= 0. ? +1.F : -1.F; // not used for Strips

            hTIDResXPrime->Fill(uOrientation * resX * cmToUm);
            hTIDResXPull->Fill(pullX);

            // update residuals
            this->updateOnlineMomenta(resDetailsTID_, detid_db, uOrientation * resX * cmToUm, pullX);

          } else if (subid == StripSubdetector::TEC) {
            uOrientation = deltaPhi(gUDirection.barePhi(), gPModule.barePhi()) >= 0. ? +1.F : -1.F;
            //vOrientation = gVDirection.perp() - gPModule.perp() >= 0. ? +1.F : -1.F; // not used for Strips

            hTECResXPrime->Fill(uOrientation * resX * cmToUm);
            hTECResXPull->Fill(pullX);

            // update residuals
            this->updateOnlineMomenta(resDetailsTEC_, detid_db, uOrientation * resX * cmToUm, pullX);
          }
        }

        const SiPixelRecHit *pixhit = dynamic_cast<const SiPixelRecHit *>(*iHit);

        if (pixhit) {
          if (pixhit->isValid()) {
            if (phase_ == SiPixelPI::phase::zero) {
              pmap->fill(detid_db, 1);
            }

            LocalPoint lp = (*iHit)->localPosition();
            //LocalError le = (*iHit)->localPositionError();
            GlobalPoint GP = geomDet->surface().toGlobal(lp);

            if ((subid == PixelSubdetector::PixelBarrel) || (subid == PixelSubdetector::PixelEndcap)) {
              // 1 = PXB, 2 = PXF
              if (subid == PixelSubdetector::PixelBarrel) {
                int layer_num = trackerTopology_->pxbLayer(detid_db);

                uOrientation = deltaPhi(gUDirection.barePhi(), gPModule.barePhi()) >= 0. ? +1.F : -1.F;
                vOrientation = gVDirection.z() - gPModule.z() >= 0 ? +1.F : -1.F;

                // if the detid has never occcurred yet, set the local orientations
                if (resDetailsBPixX_.find(detid_db) == resDetailsBPixX_.end()) {
                  resDetailsBPixX_[detid_db].rDirection = gWDirection.perp() - gPModule.perp() >= 0 ? +1 : -1;
                  resDetailsBPixX_[detid_db].zDirection = gVDirection.z() - gPModule.z() >= 0 ? +1 : -1;
                  resDetailsBPixX_[detid_db].rOrZDirection =
                      resDetailsBPixX_[detid_db].rDirection;  // barrel (split in r)
                }

                // if the detid has never occcurred yet, set the local orientations
                if (resDetailsBPixY_.find(detid_db) == resDetailsBPixY_.end()) {
                  resDetailsBPixY_[detid_db].rDirection = gWDirection.perp() - gPModule.perp() >= 0 ? +1 : -1;
                  resDetailsBPixY_[detid_db].zDirection = gVDirection.z() - gPModule.z() >= 0 ? +1 : -1;
                  resDetailsBPixY_[detid_db].rOrZDirection =
                      resDetailsBPixY_[detid_db].rDirection;  // barrel (split in r)
                }

                hHitCountVsThetaBPix->Fill(GP.theta());
                hHitCountVsPhiBPix->Fill(GP.phi());

                hHitCountVsZBPix->Fill(GP.z());
                hHitCountVsXBPix->Fill(GP.x());
                hHitCountVsYBPix->Fill(GP.y());

                hBPixResXPrime->Fill(uOrientation * resX * cmToUm);
                hBPixResYPrime->Fill(vOrientation * resY * cmToUm);
                hBPixResXPull->Fill(pullX);
                hBPixResYPull->Fill(pullY);

                if (DEBUG)
                  edm::LogVerbatim("DMRChecker") << "layer: " << layer_num << std::endl;

                // update residuals X
                this->updateOnlineMomenta(resDetailsBPixX_, detid_db, uOrientation * resX * cmToUm, pullX);

                // update residuals Y
                this->updateOnlineMomenta(resDetailsBPixY_, detid_db, vOrientation * resY * cmToUm, pullY);

                fillByIndex(barrelLayersResidualsX, layer_num, uOrientation * resX * cmToUm);
                fillByIndex(barrelLayersPullsX, layer_num, pullX);
                fillByIndex(barrelLayersResidualsY, layer_num, vOrientation * resY * cmToUm);
                fillByIndex(barrelLayersPullsY, layer_num, pullY);

              } else if (subid == PixelSubdetector::PixelEndcap) {
                uOrientation = gUDirection.perp() - gPModule.perp() >= 0 ? +1.F : -1.F;
                vOrientation = deltaPhi(gVDirection.barePhi(), gPModule.barePhi()) >= 0. ? +1.F : -1.F;

                int side_num = trackerTopology_->pxfSide(detid_db);
                int disk_num = trackerTopology_->pxfDisk(detid_db);

                int packedTopo = disk_num + 3 * (side_num - 1);

                if (DEBUG)
                  edm::LogVerbatim("DMRChecker") << "side: " << side_num << " disk: " << disk_num
                                                 << " packedTopo: " << packedTopo << " GP.z(): " << GP.z() << std::endl;

                hHitCountVsThetaFPix->Fill(GP.theta());
                hHitCountVsPhiFPix->Fill(GP.phi());

                hHitCountVsZFPix->Fill(GP.z());
                hHitCountVsXFPix->Fill(GP.x());
                hHitCountVsYFPix->Fill(GP.y());

                hFPixResXPrime->Fill(uOrientation * resX * cmToUm);
                hFPixResYPrime->Fill(vOrientation * resY * cmToUm);
                hFPixResXPull->Fill(pullX);
                hFPixResYPull->Fill(pullY);

                fillByIndex(endcapDisksResidualsX, packedTopo, uOrientation * resX * cmToUm);
                fillByIndex(endcapDisksPullsX, packedTopo, pullX);
                fillByIndex(endcapDisksResidualsY, packedTopo, vOrientation * resY * cmToUm);
                fillByIndex(endcapDisksPullsY, packedTopo, pullY);

                // if the detid has never occcurred yet, set the local orientations
                if (resDetailsFPixX_.find(detid_db) == resDetailsFPixX_.end()) {
                  resDetailsFPixX_[detid_db].rDirection = gUDirection.perp() - gPModule.perp() >= 0 ? +1 : -1;
                  resDetailsFPixX_[detid_db].zDirection = gWDirection.z() - gPModule.z() >= 0 ? +1 : -1;
                  resDetailsFPixX_[detid_db].rOrZDirection =
                      resDetailsFPixX_[detid_db].zDirection;  // endcaps (split in z)
                }

                // if the detid has never occcurred yet, set the local orientations
                if (resDetailsFPixY_.find(detid_db) == resDetailsFPixY_.end()) {
                  resDetailsFPixY_[detid_db].rDirection = gUDirection.perp() - gPModule.perp() >= 0 ? +1 : -1;
                  resDetailsFPixY_[detid_db].zDirection = gWDirection.z() - gPModule.z() >= 0 ? +1 : -1;
                  resDetailsFPixY_[detid_db].rOrZDirection =
                      resDetailsFPixY_[detid_db].zDirection;  // endcaps (split in z)
                }

                // update residuals X
                this->updateOnlineMomenta(resDetailsFPixX_, detid_db, uOrientation * resX * cmToUm, pullX);

                // update residuals Y
                this->updateOnlineMomenta(resDetailsFPixY_, detid_db, vOrientation * resY * cmToUm, pullY);

                if (side_num == 1) {
                  hHitCountVsXFPixMinus->Fill(GP.x());
                  hHitCountVsYFPixMinus->Fill(GP.y());
                  hHitCountVsZFPixMinus->Fill(GP.z());
                  hHitCountVsThetaFPixMinus->Fill(GP.theta());
                  hHitCountVsPhiFPixMinus->Fill(GP.phi());

                  hFPixZMinusResXPrime->Fill(uOrientation * resX * cmToUm);
                  hFPixZMinusResYPrime->Fill(vOrientation * resY * cmToUm);
                  hFPixZMinusResXPull->Fill(pullX);
                  hFPixZMinusResYPull->Fill(pullY);

                } else {
                  hHitCountVsXFPixPlus->Fill(GP.x());
                  hHitCountVsYFPixPlus->Fill(GP.y());
                  hHitCountVsZFPixPlus->Fill(GP.z());
                  hHitCountVsThetaFPixPlus->Fill(GP.theta());
                  hHitCountVsPhiFPixPlus->Fill(GP.phi());

                  hFPixZPlusResXPrime->Fill(uOrientation * resX * cmToUm);
                  hFPixZPlusResYPrime->Fill(vOrientation * resY * cmToUm);
                  hFPixZPlusResXPull->Fill(pullX);
                  hFPixZPlusResYPull->Fill(pullY);
                }
              }
            }
          }
        }
      }

      hHit2D->Fill(nHit2D);
      hHit->Fill(track.numberOfValidHits());
      hnhpxb->Fill(track.hitPattern().numberOfValidPixelBarrelHits());
      hnhpxe->Fill(track.hitPattern().numberOfValidPixelEndcapHits());
      hnhTIB->Fill(track.hitPattern().numberOfValidStripTIBHits());
      hnhTID->Fill(track.hitPattern().numberOfValidStripTIDHits());
      hnhTOB->Fill(track.hitPattern().numberOfValidStripTOBHits());
      hnhTEC->Fill(track.hitPattern().numberOfValidStripTECHits());

      runHitsMap_[event.run()][0] += track.hitPattern().numberOfValidPixelBarrelHits();
      runHitsMap_[event.run()][1] += track.hitPattern().numberOfValidPixelEndcapHits();
      runHitsMap_[event.run()][2] += track.hitPattern().numberOfValidStripTIBHits();
      runHitsMap_[event.run()][3] += track.hitPattern().numberOfValidStripTIDHits();
      runHitsMap_[event.run()][4] += track.hitPattern().numberOfValidStripTOBHits();
      runHitsMap_[event.run()][5] += track.hitPattern().numberOfValidStripTECHits();

      // fill hit composition histogram
      if (track.hitPattern().numberOfValidPixelBarrelHits() != 0) {
        hHitComposition->Fill(0., track.hitPattern().numberOfValidPixelBarrelHits());

        pNBpixHitsVsVx->Fill(track.vx(), track.hitPattern().numberOfValidPixelBarrelHits());
        pNBpixHitsVsVy->Fill(track.vy(), track.hitPattern().numberOfValidPixelBarrelHits());
        pNBpixHitsVsVz->Fill(track.vz(), track.hitPattern().numberOfValidPixelBarrelHits());
      }
      if (track.hitPattern().numberOfValidPixelEndcapHits() != 0) {
        hHitComposition->Fill(1., track.hitPattern().numberOfValidPixelEndcapHits());
      }
      if (track.hitPattern().numberOfValidStripTIBHits() != 0) {
        hHitComposition->Fill(2., track.hitPattern().numberOfValidStripTIBHits());
      }
      if (track.hitPattern().numberOfValidStripTIDHits() != 0) {
        hHitComposition->Fill(3., track.hitPattern().numberOfValidStripTIDHits());
      }
      if (track.hitPattern().numberOfValidStripTOBHits() != 0) {
        hHitComposition->Fill(4., track.hitPattern().numberOfValidStripTOBHits());
      }
      if (track.hitPattern().numberOfValidStripTECHits() != 0) {
        hHitComposition->Fill(5., track.hitPattern().numberOfValidStripTECHits());
      }

      hCharge->Fill(track.charge());
      hQoverP->Fill(track.qoverp());
      hQoverPZoom->Fill(track.qoverp());
      hPt->Fill(track.pt());
      hP->Fill(track.p());
      hchi2ndof->Fill(track.normalizedChi2());
      hEta->Fill(track.eta());
      hPhi->Fill(track.phi());

      if (fabs(track.eta()) < 0.8)
        hPhiBarrel->Fill(track.phi());
      if (track.eta() > 0.8 && track.eta() < 1.4)
        hPhiOverlapPlus->Fill(track.phi());
      if (track.eta() < -0.8 && track.eta() > -1.4)
        hPhiOverlapMinus->Fill(track.phi());
      if (track.eta() > 1.4)
        hPhiEndcapPlus->Fill(track.phi());
      if (track.eta() < -1.4)
        hPhiEndcapMinus->Fill(track.phi());

      hd0->Fill(track.d0());
      hdz->Fill(track.dz());
      hdxy->Fill(track.dxy());
      hvx->Fill(track.vx());
      hvy->Fill(track.vy());
      hvz->Fill(track.vz());

      htrkAlgo->Fill(track.algo());

      int myquality = -99;
      if (track.quality(reco::TrackBase::undefQuality)) {
        myquality = -1;
        htrkQuality->Fill(myquality);
      }
      if (track.quality(reco::TrackBase::loose)) {
        myquality = 0;
        htrkQuality->Fill(myquality);
      }
      if (track.quality(reco::TrackBase::tight)) {
        myquality = 1;
        htrkQuality->Fill(myquality);
      }
      if (track.quality(reco::TrackBase::highPurity) && (!isCosmics_)) {
        myquality = 2;
        htrkQuality->Fill(myquality);
        hPhp->Fill(track.p());
        hPthp->Fill(track.pt());
        hHithp->Fill(track.numberOfValidHits());
        hEtahp->Fill(track.eta());
        hPhihp->Fill(track.phi());
        hchi2ndofhp->Fill(track.normalizedChi2());
        hchi2Probhp->Fill(TMath::Prob(track.chi2(), track.ndof()));
        nHighPurityTracks++;
      }
      if (track.quality(reco::TrackBase::confirmed)) {
        myquality = 3;
        htrkQuality->Fill(myquality);
      }
      if (track.quality(reco::TrackBase::goodIterative)) {
        myquality = 4;
        htrkQuality->Fill(myquality);
      }

      // Fill 1D track histos
      static const int etaindex = this->index(vTrackHistos_, "h_tracketa");
      vTrackHistos_[etaindex]->Fill(track.eta());
      static const int phiindex = this->index(vTrackHistos_, "h_trackphi");
      vTrackHistos_[phiindex]->Fill(track.phi());
      static const int numOfValidHitsindex = this->index(vTrackHistos_, "h_trackNumberOfValidHits");
      vTrackHistos_[numOfValidHitsindex]->Fill(track.numberOfValidHits());
      static const int numOfLostHitsindex = this->index(vTrackHistos_, "h_trackNumberOfLostHits");
      vTrackHistos_[numOfLostHitsindex]->Fill(track.numberOfLostHits());

      GlobalPoint gPoint(track.vx(), track.vy(), track.vz());
      double theLocalMagFieldInInverseGeV = magneticField_->inInverseGeV(gPoint).z();
      double kappa = -track.charge() * theLocalMagFieldInInverseGeV / track.pt();

      static const int kappaindex = this->index(vTrackHistos_, "h_curvature");
      vTrackHistos_[kappaindex]->Fill(kappa);
      static const int kappaposindex = this->index(vTrackHistos_, "h_curvature_pos");
      if (track.charge() > 0)
        vTrackHistos_[kappaposindex]->Fill(fabs(kappa));
      static const int kappanegindex = this->index(vTrackHistos_, "h_curvature_neg");
      if (track.charge() < 0)
        vTrackHistos_[kappanegindex]->Fill(fabs(kappa));

      double chi2Prob = TMath::Prob(track.chi2(), track.ndof());
      double normchi2 = track.normalizedChi2();

      static const int normchi2index = this->index(vTrackHistos_, "h_normchi2");
      vTrackHistos_[normchi2index]->Fill(normchi2);
      static const int chi2index = this->index(vTrackHistos_, "h_chi2");
      vTrackHistos_[chi2index]->Fill(track.chi2());
      static const int chi2Probindex = this->index(vTrackHistos_, "h_chi2Prob");
      vTrackHistos_[chi2Probindex]->Fill(chi2Prob);
      static const int ptindex = this->index(vTrackHistos_, "h_pt");
      static const int pt2index = this->index(vTrackHistos_, "h_ptrebin");
      vTrackHistos_[ptindex]->Fill(track.pt());
      vTrackHistos_[pt2index]->Fill(track.pt());
      if (track.ptError() != 0.) {
        static const int ptResolutionindex = this->index(vTrackHistos_, "h_ptResolution");
        vTrackHistos_[ptResolutionindex]->Fill(track.ptError() / track.pt());
      }
      // Fill track profiles
      static const int d0phiindex = this->index(vTrackProfiles_, "p_d0_vs_phi");
      vTrackProfiles_[d0phiindex]->Fill(track.phi(), track.d0());
      static const int dzphiindex = this->index(vTrackProfiles_, "p_dz_vs_phi");
      vTrackProfiles_[dzphiindex]->Fill(track.phi(), track.dz());
      static const int d0etaindex = this->index(vTrackProfiles_, "p_d0_vs_eta");
      vTrackProfiles_[d0etaindex]->Fill(track.eta(), track.d0());
      static const int dzetaindex = this->index(vTrackProfiles_, "p_dz_vs_eta");
      vTrackProfiles_[dzetaindex]->Fill(track.eta(), track.dz());
      static const int chiProbphiindex = this->index(vTrackProfiles_, "p_chi2Prob_vs_phi");
      vTrackProfiles_[chiProbphiindex]->Fill(track.phi(), chi2Prob);
      static const int chiProbabsd0index = this->index(vTrackProfiles_, "p_chi2Prob_vs_d0");
      vTrackProfiles_[chiProbabsd0index]->Fill(fabs(track.d0()), chi2Prob);
      static const int chiProbabsdzindex = this->index(vTrackProfiles_, "p_chi2Prob_vs_dz");
      vTrackProfiles_[chiProbabsdzindex]->Fill(track.dz(), chi2Prob);
      static const int chiphiindex = this->index(vTrackProfiles_, "p_chi2_vs_phi");
      vTrackProfiles_[chiphiindex]->Fill(track.phi(), track.chi2());
      static const int normchiphiindex = this->index(vTrackProfiles_, "p_normchi2_vs_phi");
      vTrackProfiles_[normchiphiindex]->Fill(track.phi(), normchi2);
      static const int chietaindex = this->index(vTrackProfiles_, "p_chi2_vs_eta");
      vTrackProfiles_[chietaindex]->Fill(track.eta(), track.chi2());
      static const int normchiptindex = this->index(vTrackProfiles_, "p_normchi2_vs_pt");
      vTrackProfiles_[normchiptindex]->Fill(track.pt(), normchi2);
      static const int normchipindex = this->index(vTrackProfiles_, "p_normchi2_vs_p");
      vTrackProfiles_[normchipindex]->Fill(track.p(), normchi2);
      static const int chiProbetaindex = this->index(vTrackProfiles_, "p_chi2Prob_vs_eta");
      vTrackProfiles_[chiProbetaindex]->Fill(track.eta(), chi2Prob);
      static const int normchietaindex = this->index(vTrackProfiles_, "p_normchi2_vs_eta");
      vTrackProfiles_[normchietaindex]->Fill(track.eta(), normchi2);
      static const int kappaphiindex = this->index(vTrackProfiles_, "p_kappa_vs_phi");
      vTrackProfiles_[kappaphiindex]->Fill(track.phi(), kappa);
      static const int kappaetaindex = this->index(vTrackProfiles_, "p_kappa_vs_eta");
      vTrackProfiles_[kappaetaindex]->Fill(track.eta(), kappa);
      static const int ptResphiindex = this->index(vTrackProfiles_, "p_ptResolution_vs_phi");
      vTrackProfiles_[ptResphiindex]->Fill(track.phi(), track.ptError() / track.pt());
      static const int ptResetaindex = this->index(vTrackProfiles_, "p_ptResolution_vs_eta");
      vTrackProfiles_[ptResetaindex]->Fill(track.eta(), track.ptError() / track.pt());

      // Fill 2D track histos
      static const int etaphiindex_2d = this->index(vTrack2DHistos_, "h2_phi_vs_eta");
      vTrack2DHistos_[etaphiindex_2d]->Fill(track.eta(), track.phi());
      static const int d0phiindex_2d = this->index(vTrack2DHistos_, "h2_d0_vs_phi");
      vTrack2DHistos_[d0phiindex_2d]->Fill(track.phi(), track.d0());
      static const int dzphiindex_2d = this->index(vTrack2DHistos_, "h2_dz_vs_phi");
      vTrack2DHistos_[dzphiindex_2d]->Fill(track.phi(), track.dz());
      static const int d0etaindex_2d = this->index(vTrack2DHistos_, "h2_d0_vs_eta");
      vTrack2DHistos_[d0etaindex_2d]->Fill(track.eta(), track.d0());
      static const int dzetaindex_2d = this->index(vTrack2DHistos_, "h2_dz_vs_eta");
      vTrack2DHistos_[dzetaindex_2d]->Fill(track.eta(), track.dz());
      static const int chiphiindex_2d = this->index(vTrack2DHistos_, "h2_chi2_vs_phi");
      vTrack2DHistos_[chiphiindex_2d]->Fill(track.phi(), track.chi2());
      static const int chiProbphiindex_2d = this->index(vTrack2DHistos_, "h2_chi2Prob_vs_phi");
      vTrack2DHistos_[chiProbphiindex_2d]->Fill(track.phi(), chi2Prob);
      static const int chiProbabsd0index_2d = this->index(vTrack2DHistos_, "h2_chi2Prob_vs_d0");
      vTrack2DHistos_[chiProbabsd0index_2d]->Fill(fabs(track.d0()), chi2Prob);
      static const int normchiphiindex_2d = this->index(vTrack2DHistos_, "h2_normchi2_vs_phi");
      vTrack2DHistos_[normchiphiindex_2d]->Fill(track.phi(), normchi2);
      static const int chietaindex_2d = this->index(vTrack2DHistos_, "h2_chi2_vs_eta");
      vTrack2DHistos_[chietaindex_2d]->Fill(track.eta(), track.chi2());
      static const int chiProbetaindex_2d = this->index(vTrack2DHistos_, "h2_chi2Prob_vs_eta");
      vTrack2DHistos_[chiProbetaindex_2d]->Fill(track.eta(), chi2Prob);
      static const int normchietaindex_2d = this->index(vTrack2DHistos_, "h2_normchi2_vs_eta");
      vTrack2DHistos_[normchietaindex_2d]->Fill(track.eta(), normchi2);
      static const int kappaphiindex_2d = this->index(vTrack2DHistos_, "h2_kappa_vs_phi");
      vTrack2DHistos_[kappaphiindex_2d]->Fill(track.phi(), kappa);
      static const int kappaetaindex_2d = this->index(vTrack2DHistos_, "h2_kappa_vs_eta");
      vTrack2DHistos_[kappaetaindex_2d]->Fill(track.eta(), kappa);
      static const int normchi2kappa_2d = this->index(vTrack2DHistos_, "h2_normchi2_vs_kappa");
      vTrack2DHistos_[normchi2kappa_2d]->Fill(normchi2, kappa);

      if (DEBUG)
        edm::LogVerbatim("DMRChecker") << "filling histos" << std::endl;

      //dxy with respect to the beamspot
      reco::BeamSpot beamSpot;
      edm::Handle<reco::BeamSpot> beamSpotHandle = event.getHandle(beamspotToken_);
      if (beamSpotHandle.isValid()) {
        beamSpot = *beamSpotHandle;
        math::XYZPoint point(beamSpot.x0(), beamSpot.y0(), beamSpot.z0());
        double dxy = track.dxy(point);
        double dz = track.dz(point);
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
        for (reco::VertexCollection::const_iterator pvtx = vertexHandle->begin(); pvtx != vertexHandle->end(); ++pvtx) {
          math::XYZPoint mypoint(pvtx->x(), pvtx->y(), pvtx->z());
          if (abs(mindxy) > abs(track.dxy(mypoint))) {
            mindxy = track.dxy(mypoint);
            dz = track.dz(mypoint);
            if (DEBUG)
              edm::LogVerbatim("DMRChecker") << "dxy: " << mindxy << " dz: " << dz << std::endl;
          }
        }

        hdxyPV->Fill(mindxy);
        hd0PV->Fill(-mindxy);
        hdzPV->Fill(dz);

        hd0PVvsphi->Fill(track.phi(), -mindxy);
        hd0PVvseta->Fill(track.eta(), -mindxy);
        hd0PVvspt->Fill(track.pt(), -mindxy);

      } else {
        hdxyPV->Fill(100);
        hd0PV->Fill(100);
        hdzPV->Fill(100);
      }

      if (DEBUG)
        edm::LogVerbatim("DMRChecker") << "end of track loop" << std::endl;
    }

    if (DEBUG)
      edm::LogVerbatim("DMRChecker") << "end of analysis" << std::endl;

    hNtrk->Fill(tC.size());
    hNtrkZoom->Fill(tC.size());
    hNhighPurity->Fill(nHighPurityTracks);

    if (DEBUG)
      edm::LogVerbatim("DMRChecker") << "end of analysis" << std::endl;
  }

  //*************************************************************
  void beginRun(edm::Run const &run, edm::EventSetup const &setup) override
  //*************************************************************
  {
    // initialize runInfoMap_
    runInfoMap_[run.run()].first = 0;
    runInfoMap_[run.run()].second = 0;

    // initialize runHitsMap
    for (int n : {0, 1, 2, 3, 4, 5}) {
      runHitsMap_[run.run()][n] = 0;  // 6 subdets
    }

    // Magnetic Field setup
    magneticField_ = &setup.getData(magFieldToken_);
    float B_ = magneticField_->inTesla(GlobalPoint(0, 0, 0)).mag();

    edm::LogInfo("DMRChecker") << "run number:" << run.run() << " magnetic field: " << B_ << " [T]" << std::endl;

    const RunInfo *summary = &setup.getData(runInfoToken_);
    time_t start_time = summary->m_start_time_ll;
    ctime(&start_time);
    time_t end_time = summary->m_stop_time_ll;
    ctime(&end_time);

    float average_current = summary->m_avg_current;
    float uptimeInSeconds = summary->m_run_intervall_micros;
    edm::LogVerbatim("DMRChecker") << " start_time: " << start_time << " ( " << summary->m_start_time_str << " )"
                                   << " | end_time: " << end_time << " ( " << summary->m_stop_time_str << " )"
                                   << " | average current: " << average_current
                                   << " | uptime in seconds: " << uptimeInSeconds << std::endl;

    double seconds = difftime(end_time, start_time) / 1.0e+6;  // convert from micros-seconds
    edm::LogVerbatim("DMRChecker") << "time difference: " << seconds << " s" << std::endl;
    timeMap_[run.run()] = seconds;

    //SiStrip Latency
    const SiStripLatency *apvlat = &setup.getData(latencyToken_);
    if (apvlat->singleReadOutMode() == 1) {
      mode = 1;  // peak mode
    } else if (apvlat->singleReadOutMode() == 0) {
      mode = -1;  // deco mode
    }

    conditionsMap_[run.run()].first = mode;
    conditionsMap_[run.run()].second = B_;

    // set geometry and topology
    trackerGeometry_ = &setup.getData(geomToken_);
    trackerTopology_ = &setup.getData(topoToken_);
  }

  //*************************************************************
  void endRun(edm::Run const &, edm::EventSetup const &) override {}
  //*************************************************************

  void beginJob() override {
    if (DEBUG)
      edm::LogVerbatim("DMRChecker") << __LINE__ << endl;

    TH1D::SetDefaultSumw2(kTRUE);

    etaMax_ = 3.;  // assign max value to eta

    // intialize counters
    ievt = 0;
    itrks = 0;

    hrun = book<TH1D>("h_run", "run", 100000, 230000, 240000);
    hlumi = book<TH1D>("h_lumi", "lumi", 1000, 0, 1000);

    // clang-format off

    hchi2ndof = book<TH1D>("h_chi2ndof", "chi2/ndf;#chi^{2}/ndf;tracks", 100, 0, 5.);
    hCharge = book<TH1D>("h_charge", "charge;Charge of the track;tracks", 5, -2.5, 2.5);
    hNtrk = book<TH1D>("h_Ntrk", "ntracks;Number of Tracks;events", 200, 0., 200.);
    hNtrkZoom = book<TH1D>("h_NtrkZoom", "Number of tracks; number of tracks;events", 10, 0., 10.);
    hNhighPurity = book<TH1D>("h_NhighPurity", "n. high purity tracks;Number of high purity tracks;events", 200, 0., 200.);

    int nAlgos = reco::TrackBase::algoSize;
    htrkAlgo = book<TH1I>("h_trkAlgo", "tracking step;iterative tracking step;tracks", nAlgos, -0.5, nAlgos - 0.5);
    for (int nbin = 1; nbin <= htrkAlgo->GetNbinsX(); nbin++) {
      htrkAlgo->GetXaxis()->SetBinLabel(nbin, reco::TrackBase::algoNames[nbin - 1].c_str());
    }

    htrkQuality = book<TH1I>("h_trkQuality", "track quality;track quality;tracks", 6, -1, 5);
    std::string qualities[7] = {"undef", "loose", "tight", "highPurity", "confirmed", "goodIterative"};
    for (int nbin = 1; nbin <= htrkQuality->GetNbinsX(); nbin++) {
      htrkQuality->GetXaxis()->SetBinLabel(nbin, qualities[nbin - 1].c_str());
    }

    hP = book<TH1D>("h_P", "Momentum;track momentum [GeV];tracks", 100, 0., 100.);
    hQoverP = book<TH1D>("h_qoverp", "Track q/p; track q/p [GeV^{-1}];tracks", 100, -1., 1.);
    hQoverPZoom = book<TH1D>("h_qoverpZoom", "Track q/p; track q/p [GeV^{-1}];tracks", 100, -0.1, 0.1);
    hPt = book<TH1D>("h_Pt", "Transverse Momentum;track p_{T} [GeV];tracks", 100, 0., 100.);
    hHit = book<TH1D>("h_nHits", "Number of hits;track n. hits;tracks", 50, -0.5, 49.5);
    hHit2D = book<TH1D>("h_nHit2D", "Number of 2D hits; number of 2D hits;tracks", 20, 0, 20);

    // Pixel

    hBPixResXPrime = book<TH1D>("h_BPixResXPrime", "BPix track X-residuals;res_{X'} [#mum];hits", 100, -1000., 1000.);
    hFPixResXPrime = book<TH1D>("h_FPixResXPrime", "FPix track X-residuals;res_{X'} [#mum];hits", 100, -1000., 1000.);
    hFPixZPlusResXPrime = book<TH1D>("h_FPixZPlusResXPrime", "FPix (Z+) track X-residuals;res_{X'} [#mum];hits", 100, -1000., 1000.);
    hFPixZMinusResXPrime = book<TH1D>("h_FPixZMinusResXPrime", "FPix (Z-) track X-residuals;res_{X'} [#mum];hits", 100, -1000., 1000.);

    hBPixResYPrime = book<TH1D>("h_BPixResYPrime", "BPix track Y-residuals;res_{Y'} [#mum];hits", 100, -1000., 1000.);
    hFPixResYPrime = book<TH1D>("h_FPixResYPrime", "FPix track Y-residuals;res_{Y'} [#mum];hits", 100, -1000., 1000.);
    hFPixZPlusResYPrime = book<TH1D>("h_FPixZPlusResYPrime", "FPix (Z+) track Y-residuals;res_{Y'} [#mum];hits", 100, -1000., 1000.);
    hFPixZMinusResYPrime = book<TH1D>("h_FPixZMinusResYPrime", "FPix (Z-) track Y-residuals;res_{Y'} [#mum];hits", 100, -1000., 1000.);

    hBPixResXPull = book<TH1D>("h_BPixResXPull", "BPix track X-pulls;res_{X'}/#sigma_{res_{X'}};hits", 100, -5., 5.);
    hFPixResXPull = book<TH1D>("h_FPixResXPull", "FPix track X-pulls;res_{X'}/#sigma_{res_{X'}};hits", 100, -5., 5.);
    hFPixZPlusResXPull = book<TH1D>("h_FPixZPlusResXPull", "FPix (Z+) track X-pulls;res_{X'}/#sigma_{res_{X'}};hits", 100, -5., 5.);
    hFPixZMinusResXPull = book<TH1D>("h_FPixZMinusResXPull", "FPix (Z-) track X-pulls;res_{X'}/#sigma_{res_{X'}};hits", 100, -5., 5.);

    hBPixResYPull = book<TH1D>("h_BPixResYPull", "BPix track Y-pulls;res_{Y'}/#sigma_{res_{Y'}};hits", 100, -5., 5.);
    hFPixResYPull = book<TH1D>("h_FPixResYPull", "FPix track Y-pulls;res_{Y'}/#sigma_{res_{Y'}};hits", 100, -5., 5.);
    hFPixZPlusResYPull = book<TH1D>("h_FPixZPlusResYPull", "FPix (Z+) track Y-pulls;res_{Y'}/#sigma_{res_{Y'}};hits", 100, -5., 5.);
    hFPixZMinusResYPull = book<TH1D>("h_FPixZMinusResYPull", "FPix (Z-) track Y-pulls;res_{Y'}/#sigma_{res_{Y'}};hits", 100, -5., 5.);

    // Strips

    hTIBResXPrime = book<TH1D>("h_TIBResXPrime", "TIB track X-residuals;res_{X'} [#mum];hits", 100, -1000., 1000.);
    hTOBResXPrime = book<TH1D>("h_TOBResXPrime", "TOB track X-residuals;res_{X'} [#mum];hits", 100, -1000., 1000.);
    hTIDResXPrime = book<TH1D>("h_TIDResXPrime", "TID track X-residuals;res_{X'} [#mum];hits", 100, -1000., 1000.);
    hTECResXPrime = book<TH1D>("h_TECResXPrime", "TEC track X-residuals;res_{X'} [#mum];hits", 100, -1000., 1000.);

    hTIBResXPull = book<TH1D>("h_TIBResXPull", "TIB track X-pulls;res_{X'}/#sigma_{res_{X'}};hits", 100, -5., 5.);
    hTOBResXPull = book<TH1D>("h_TOBResXPull", "TOB track X-pulls;res_{X'}/#sigma_{res_{X'}};hits", 100, -5., 5.);
    hTIDResXPull = book<TH1D>("h_TIDResXPull", "TID track X-pulls;res_{X'}/#sigma_{res_{X'}};hits", 100, -5., 5.);
    hTECResXPull = book<TH1D>("h_TECResXPull", "TEC track X-pulls;res_{X'}/#sigma_{res_{X'}};hits", 100, -5., 5.);

    // hit counts

    hHitCountVsZBPix = book<TH1D>("h_HitCountVsZBpix", "Number of BPix hits vs z;hit global z;hits", 60, -30, 30);
    hHitCountVsZFPix = book<TH1D>("h_HitCountVsZFpix", "Number of FPix hits vs z;hit global z;hits", 100, -100, 100);

    hHitCountVsXBPix = book<TH1D>("h_HitCountVsXBpix", "Number of BPix hits vs x;hit global x;hits", 20, -20, 20);
    hHitCountVsXFPix = book<TH1D>("h_HitCountVsXFpix", "Number of FPix hits vs x;hit global x;hits", 20, -20, 20);

    hHitCountVsYBPix = book<TH1D>("h_HitCountVsYBpix", "Number of BPix hits vs y;hit global y;hits", 20, -20, 20);
    hHitCountVsYFPix = book<TH1D>("h_HitCountVsYFpix", "Number of FPix hits vs y;hit global y;hits", 20, -20, 20);

    hHitCountVsThetaBPix = book<TH1D>("h_HitCountVsThetaBpix", "Number of BPix hits vs #theta;hit global #theta;hits", 20, 0., M_PI);
    hHitCountVsPhiBPix = book<TH1D>("h_HitCountVsPhiBpix", "Number of BPix hits vs #phi;hit global #phi;hits", 20, -M_PI, M_PI);

    hHitCountVsThetaFPix = book<TH1D>("h_HitCountVsThetaFpix", "Number of FPix hits vs #theta;hit global #theta;hits", 40, 0., M_PI);
    hHitCountVsPhiFPix = book<TH1D>("h_HitCountVsPhiFpix", "Number of FPix hits vs #phi;hit global #phi;hits", 20, -M_PI, M_PI);

    // two sides of FPix

    hHitCountVsZFPixPlus = book<TH1D>("h_HitCountVsZFPixPlus", "Number of FPix(Z+) hits vs z;hit global z;hits", 60, 15., 60);
    hHitCountVsZFPixMinus = book<TH1D>("h_HitCountVsZFPixMinus", "Number of FPix(Z-) hits vs z;hit global z;hits", 100, -60., -15.);

    hHitCountVsXFPixPlus = book<TH1D>("h_HitCountVsXFPixPlus", "Number of FPix(Z+) hits vs x;hit global x;hits", 20, -20, 20);
    hHitCountVsXFPixMinus = book<TH1D>("h_HitCountVsXFPixMinus", "Number of FPix(Z-) hits vs x;hit global x;hits", 20, -20, 20);

    hHitCountVsYFPixPlus = book<TH1D>("h_HitCountVsYFPixPlus", "Number of FPix(Z+) hits vs y;hit global y;hits", 20, -20, 20);
    hHitCountVsYFPixMinus = book<TH1D>("h_HitCountVsYFPixMinus", "Number of FPix(Z-) hits vs y;hit global y;hits", 20, -20, 20);

    hHitCountVsThetaFPixPlus = book<TH1D>("h_HitCountVsThetaFPixPlus", "Number of FPix(Z+) hits vs #theta;hit global #theta;hits", 20, 0., M_PI);
    hHitCountVsPhiFPixPlus = book<TH1D>("h_HitCountVsPhiFPixPlus","Number of FPix(Z+) hits vs #phi;hit global #phi;hits",20,-M_PI,M_PI);

    hHitCountVsThetaFPixMinus = book<TH1D>("h_HitCountVsThetaFPixMinus", "Number of FPix(Z+) hits vs #theta;hit global #theta;hits", 40, 0., M_PI);
    hHitCountVsPhiFPixMinus = book<TH1D>("h_HitCountVsPhiFPixMinus","Number of FPix(Z+) hits vs #phi;hit global #phi;hits",20,-M_PI,M_PI);

    TFileDirectory ByLayerResiduals = fs->mkdir("ByLayerResiduals");
    barrelLayersResidualsX = bookResidualsHistogram(ByLayerResiduals, 4, "X", "Res", "BPix");
    endcapDisksResidualsX = bookResidualsHistogram(ByLayerResiduals, 6, "X", "Res", "FPix");
    barrelLayersResidualsY = bookResidualsHistogram(ByLayerResiduals, 4, "Y", "Res", "BPix");
    endcapDisksResidualsY = bookResidualsHistogram(ByLayerResiduals, 6, "Y", "Res", "FPix");

    TFileDirectory ByLayerPulls = fs->mkdir("ByLayerPulls");
    barrelLayersPullsX = bookResidualsHistogram(ByLayerPulls, 4, "X", "Pull", "BPix");
    endcapDisksPullsX = bookResidualsHistogram(ByLayerPulls, 6, "X", "Pull", "FPix");
    barrelLayersPullsY = bookResidualsHistogram(ByLayerPulls, 4, "Y", "Pull", "BPix");
    endcapDisksPullsY = bookResidualsHistogram(ByLayerPulls, 6, "Y", "Pull", "FPix");

    hEta = book<TH1D>("h_Eta", "Track pseudorapidity; track #eta;tracks", 100, -etaMax_, etaMax_);
    hPhi = book<TH1D>("h_Phi", "Track azimuth; track #phi;tracks", 100, -M_PI, M_PI);

    hPhiBarrel = book<TH1D>("h_PhiBarrel", "hPhiBarrel (0<|#eta|<0.8);track #Phi;tracks", 100, -M_PI, M_PI);
    hPhiOverlapPlus = book<TH1D>("h_PhiOverlapPlus", "hPhiOverlapPlus (0.8<#eta<1.4);track #phi;tracks", 100, -M_PI, M_PI);
    hPhiOverlapMinus = book<TH1D>("h_PhiOverlapMinus", "hPhiOverlapMinus (-1.4<#eta<-0.8);track #phi;tracks", 100, -M_PI, M_PI);
    hPhiEndcapPlus = book<TH1D>("h_PhiEndcapPlus", "hPhiEndcapPlus (#eta>1.4);track #phi;track", 100, -M_PI, M_PI);
    hPhiEndcapMinus = book<TH1D>("h_PhiEndcapMinus", "hPhiEndcapMinus (#eta<1.4);track #phi;tracks", 100, -M_PI, M_PI);

    if (!isCosmics_) {
      hPhp = book<TH1D>("h_P_hp", "Momentum (high purity);track momentum [GeV];tracks", 100, 0., 100.);
      hPthp = book<TH1D>("h_Pt_hp", "Transverse Momentum (high purity);track p_{T} [GeV];tracks", 100, 0., 100.);
      hHithp = book<TH1D>("h_nHit_hp", "Number of hits (high purity);track n. hits;tracks", 30, 0, 30);
      hEtahp = book<TH1D>("h_Eta_hp", "Track pseudorapidity (high purity); track #eta;tracks", 100, -etaMax_, etaMax_);
      hPhihp = book<TH1D>("h_Phi_hp", "Track azimuth (high purity); track #phi;tracks", 100, -M_PI, M_PI);
      hchi2ndofhp = book<TH1D>("h_chi2ndof_hp", "chi2/ndf (high purity);#chi^{2}/ndf;tracks", 100, 0, 5.);
      hchi2Probhp = book<TH1D>("hchi2_Prob_hp", "#chi^{2} probability (high purity);#chi^{2}prob_{Track};Number of Tracks", 100, 0.0, 1.);

      hvx = book<TH1D>("h_vx", "Track v_{x} ; track v_{x} [cm];tracks", 100, -1.5, 1.5);
      hvy = book<TH1D>("h_vy", "Track v_{y} ; track v_{y} [cm];tracks", 100, -1.5, 1.5);
      hvz = book<TH1D>("h_vz", "Track v_{z} ; track v_{z} [cm];tracks", 100, -20., 20.);
      hd0 = book<TH1D>("h_d0", "Track d_{0} ; track d_{0} [cm];tracks", 100, -1., 1.);
      hdxy = book<TH1D>("h_dxy", "Track d_{xy}; track d_{xy} [cm]; tracks", 100, -0.5, 0.5);
      hdz = book<TH1D>("h_dz", "Track d_{z} ; track d_{z} [cm]; tracks", 100, -20, 20);

      hd0PVvsphi = book<TH2D>("h2_d0PVvsphi", "hd0PVvsphi;track #phi;track d_{0}(PV) [cm]", 160, -M_PI, M_PI, 100, -1., 1.);
      hd0PVvseta = book<TH2D>("h2_d0PVvseta", "hdPV0vseta;track #eta;track d_{0}(PV) [cm]", 160, -etaMax_, etaMax_, 100, -1., 1.);
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

      hd0vsphi = book<TH2D>("h2_d0vsphi", "Track d_{0} vs #phi; track #phi;track d_{0} [cm]", 160, -M_PI, M_PI, 100, -100., 100.);
      hd0vseta = book<TH2D>("h2_d0vseta", "Track d_{0} vs #eta; track #eta;track d_{0} [cm]", 160, -etaMax_, etaMax_, 100, -100., 100.);
      hd0vspt = book<TH2D>("h2_d0vspt", "Track d_{0} vs p_{T};track p_{T};track d_{0} [cm]", 50, 0., 100., 100, -100, 100);

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
    pNBpixHitsVsVx = book<TProfile>("p_NpixHits_vs_Vx", "n. Barrel Pixel hits vs. v_{x};v_{x} (cm);n. BPix hits", 20, -20, 20);
    pNBpixHitsVsVy = book<TProfile>("p_NpixHits_vs_Vy", "n. Barrel Pixel hits vs. v_{y};v_{y} (cm);n. BPix hits", 20, -20, 20);
    pNBpixHitsVsVz = book<TProfile>("p_NpixHits_vs_Vz", "n. Barrel Pixel hits vs. v_{z};v_{z} (cm);n. BPix hits", 20, -100, 100);

    std::string dets[6] = {"PXB", "PXF", "TIB", "TID", "TOB", "TEC"};
    for (int i = 1; i <= hHitComposition->GetNbinsX(); i++) {
      hHitComposition->GetXaxis()->SetBinLabel(i, dets[i - 1].c_str());
    }

    vTrackHistos_.push_back(book<TH1F>("h_tracketa", "Track #eta;#eta_{Track};Number of Tracks", 90, -etaMax_, etaMax_));
    vTrackHistos_.push_back(book<TH1F>("h_trackphi", "Track #phi;#phi_{Track};Number of Tracks", 90, -M_PI, M_PI));
    vTrackHistos_.push_back(book<TH1F>("h_trackNumberOfValidHits", "Track # of valid hits;# of valid hits _{Track};Number of Tracks", 40, 0., 40.));
    vTrackHistos_.push_back(book<TH1F>("h_trackNumberOfLostHits", "Track # of lost hits;# of lost hits _{Track};Number of Tracks", 10, 0., 10.));
    vTrackHistos_.push_back(book<TH1F>("h_curvature", "Curvature #kappa;#kappa_{Track};Number of Tracks", 100, -.05, .05));
    vTrackHistos_.push_back(book<TH1F>("h_curvature_pos", "Curvature |#kappa| Positive Tracks;|#kappa_{pos Track}|;Number of Tracks", 100, .0, .05));
    vTrackHistos_.push_back(book<TH1F>("h_curvature_neg", "Curvature |#kappa| Negative Tracks;|#kappa_{neg Track}|;Number of Tracks", 100, .0, .05));
    vTrackHistos_.push_back(book<TH1F>("h_diff_curvature", "Curvature |#kappa| Tracks Difference;|#kappa_{Track}|;# Pos Tracks - # Neg Tracks", 100,.0,.05));

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
    vTrackProfiles_.push_back(book<TProfile>("p_chi2Prob_vs_phi","#chi^{2} probablility vs. #phi;#phi_{Track};#LT #chi^{2} probability#GT",100,-M_PI,M_PI));
    vTrackProfiles_.push_back(book<TProfile>("p_chi2Prob_vs_d0", "#chi^{2} probablility vs. |d_{0}|;|d_{0}|[cm];#LT #chi^{2} probability#GT", 100, 0, 80));
    vTrackProfiles_.push_back(book<TProfile>("p_chi2Prob_vs_dz", "#chi^{2} probablility vs. dz;d_{z} [cm];#LT #chi^{2} probability#GT", 100, -30, 30));
    vTrackProfiles_.push_back(book<TProfile>("p_normchi2_vs_phi", "#chi^{2}/ndof vs. #phi;#phi_{Track};#LT #chi^{2}/ndof #GT", 100, -M_PI, M_PI));
    vTrackProfiles_.push_back(book<TProfile>("p_chi2_vs_eta", "#chi^{2} vs. #eta;#eta_{Track};#LT #chi^{2} #GT", 100, -etaMax_, etaMax_));
    vTrackProfiles_.push_back(book<TProfile>("p_normchi2_vs_pt", "norm #chi^{2} vs. p_{T}_{Track}; p_{T}_{Track};#LT #chi^{2}/ndof #GT", 18, xBins));
    vTrackProfiles_.push_back(book<TProfile>("p_normchi2_vs_p", "#chi^{2}/ndof vs. p_{Track};p_{Track};#LT #chi^{2}/ndof #GT", 18, xBins));
    vTrackProfiles_.push_back(book<TProfile>("p_chi2Prob_vs_eta","#chi^{2} probability vs. #eta;#eta_{Track};#LT #chi^{2} probability #GT",100,-etaMax_, etaMax_));
    vTrackProfiles_.push_back(book<TProfile>("p_normchi2_vs_eta", "#chi^{2}/ndof vs. #eta;#eta_{Track};#LT #chi^{2}/ndof #GT", 100, -etaMax_, etaMax_));
    vTrackProfiles_.push_back(book<TProfile>("p_kappa_vs_phi", "#kappa vs. #phi;#phi_{Track};#kappa", 100, -M_PI, M_PI));
    vTrackProfiles_.push_back(book<TProfile>("p_kappa_vs_eta", "#kappa vs. #eta;#eta_{Track};#kappa", 100, -etaMax_, etaMax_));
    vTrackProfiles_.push_back(book<TProfile>("p_ptResolution_vs_phi","#delta_{p_{T}}/p_{T}^{track};#phi^{track};#delta_{p_{T}}/p_{T}^{track}", 100,-M_PI,M_PI));
    vTrackProfiles_.push_back(book<TProfile>("p_ptResolution_vs_eta","#delta_{p_{T}}/p_{T}^{track};#eta^{track};#delta_{p_{T}}/p_{T}^{track}", 100, -etaMax_, etaMax_));
    vTrack2DHistos_.push_back(book<TH2F>("h2_d0_vs_phi","Transverse Impact Parameter vs. #phi;#phi_{Track};d_{0} [cm]", 100, -M_PI, M_PI, 100, -1., 1.));
    vTrack2DHistos_.push_back(book<TH2F>("h2_phi_vs_eta", "Track #phi vs. #eta;#eta_{Track};#phi_{Track}",50, -etaMax_, etaMax_, 50, -M_PI, M_PI));
    vTrack2DHistos_.push_back(book<TH2F>("h2_dz_vs_phi","Longitudinal Impact Parameter vs. #phi;#phi_{Track};d_{z} [cm]",100,-M_PI,M_PI,100,-100.,100.));
    vTrack2DHistos_.push_back(book<TH2F>("h2_d0_vs_eta","Transverse Impact Parameter vs. #eta;#eta_{Track};d_{0} [cm]", 100, -etaMax_, etaMax_, 100, -1., 1.));
    vTrack2DHistos_.push_back(book<TH2F>("h2_dz_vs_eta","Longitudinal Impact Parameter vs. #eta;#eta_{Track};d_{z} [cm]",100, -etaMax_, etaMax_, 100,-100.,100.));
    vTrack2DHistos_.push_back(book<TH2F>("h2_chi2_vs_phi", "#chi^{2} vs. #phi;#phi_{Track};#chi^{2}", 100, -M_PI, M_PI, 500, 0., 500.));
    vTrack2DHistos_.push_back(book<TH2F>("h2_chi2Prob_vs_phi","#chi^{2} probability vs. #phi;#phi_{Track};#chi^{2} probability",100,-M_PI,M_PI,100,0.,1.));
    vTrack2DHistos_.push_back(book<TH2F>("h2_chi2Prob_vs_d0","#chi^{2} probability vs. |d_{0}|;|d_{0}| [cm];#chi^{2} probability",100,0,80,100,0.,1.));
    vTrack2DHistos_.push_back(book<TH2F>("h2_normchi2_vs_phi", "#chi^{2}/ndof vs. #phi;#phi_{Track};#chi^{2}/ndof", 100, -M_PI, M_PI, 100, 0., 10.));
    vTrack2DHistos_.push_back(book<TH2F>("h2_chi2_vs_eta", "#chi^{2} vs. #eta;#eta_{Track};#chi^{2}", 100, -etaMax_, etaMax_, 500, 0., 500.));
    vTrack2DHistos_.push_back(book<TH2F>("h2_chi2Prob_vs_eta","#chi^{2} probaility vs. #eta;#eta_{Track};#chi^{2} probability",100,-M_PI,M_PI,100,0.,1.));
    vTrack2DHistos_.push_back(book<TH2F>("h2_normchi2_vs_eta", "#chi^{2}/ndof vs. #eta;#eta_{Track};#chi^{2}/ndof", 100, -etaMax_, etaMax_, 100, 0., 10.));
    vTrack2DHistos_.push_back(book<TH2F>("h2_kappa_vs_phi", "#kappa vs. #phi;#phi_{Track};#kappa", 100, -M_PI, M_PI, 100, .0, .05));
    vTrack2DHistos_.push_back(book<TH2F>("h2_kappa_vs_eta", "#kappa vs. #eta;#eta_{Track};#kappa", 100, -etaMax_, etaMax_, 100, .0, .05));
    vTrack2DHistos_.push_back(book<TH2F>("h2_normchi2_vs_kappa", "#kappa vs. #chi^{2}/ndof;#chi^{2}/ndof;#kappa", 100, 0., 10, 100, -.03, .03));

    // clang-format on

    firstEvent_ = true;

    // create the full maps
    fullPixelmapXDMR->createTrackerBaseMap();
    fullPixelmapYDMR->createTrackerBaseMap();

  }  //beginJob

  void endJob() override {
    edm::LogPrint("DMRChecker") << "*******************************" << std::endl;
    edm::LogPrint("DMRChecker") << "Events run in total: " << ievt << std::endl;
    edm::LogPrint("DMRChecker") << "n. tracks: " << itrks << std::endl;
    edm::LogPrint("DMRChecker") << "*******************************" << std::endl;

    int nFiringTriggers = !triggerMap_.empty() ? triggerMap_.size() : 1;
    edm::LogPrint("DMRChecker") << "firing triggers: " << triggerMap_.size() << std::endl;
    edm::LogPrint("DMRChecker") << "*******************************" << std::endl;

    tksByTrigger_ =
        book<TH1D>("tksByTrigger", "tracks by HLT path;;% of # traks", nFiringTriggers, -0.5, nFiringTriggers - 0.5);
    evtsByTrigger_ =
        book<TH1D>("evtsByTrigger", "events by HLT path;;% of # events", nFiringTriggers, -0.5, nFiringTriggers - 0.5);

    if (DEBUG)
      edm::LogPrint("DMRChecker") << __FILE__ << "@" << __FUNCTION__ << " L-" << __LINE__ << std::endl;

    int i = 0;
    for (const auto &it : triggerMap_) {
      i++;

      double trkpercent = ((it.second).second) * 100. / double(itrks);
      double evtpercent = ((it.second).first) * 100. / double(ievt);

      std::cout.precision(4);

      edm::LogPrint("DMRChecker") << "HLT path: " << std::setw(60) << left << it.first << " | events firing: " << right
                                  << std::setw(8) << (it.second).first << " (" << setw(8) << fixed << evtpercent << "%)"
                                  << " | tracks collected: " << std::setw(10) << (it.second).second << " (" << setw(8)
                                  << fixed << trkpercent << "%)";

      tksByTrigger_->SetBinContent(i, trkpercent);
      tksByTrigger_->GetXaxis()->SetBinLabel(i, (it.first).c_str());

      evtsByTrigger_->SetBinContent(i, evtpercent);
      evtsByTrigger_->GetXaxis()->SetBinLabel(i, (it.first).c_str());
    }

    if (DEBUG)
      edm::LogPrint("DMRChecker") << __FILE__ << "@" << __FUNCTION__ << " L-" << __LINE__ << std::endl;

    int nRuns = conditionsMap_.size();
    if (nRuns < 1)
      return;

    vector<int> theRuns_;
    for (const auto &it : conditionsMap_) {
      theRuns_.push_back(it.first);
    }

    std::sort(theRuns_.begin(), theRuns_.end());
    int runRange = theRuns_.back() - theRuns_.front() + 1;

    edm::LogPrint("DMRChecker") << "*******************************" << std::endl;
    edm::LogPrint("DMRChecker") << "first run: " << theRuns_.front() << std::endl;
    edm::LogPrint("DMRChecker") << "last run:  " << theRuns_.back() << std::endl;
    edm::LogPrint("DMRChecker") << "considered runs: " << nRuns << std::endl;
    edm::LogPrint("DMRChecker") << "*******************************" << std::endl;

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

    tracksByRun_ = book<TH1D>("tracksByRun",
                              "n. AlCaReco Tracks by run number;;n. of tracks",
                              runRange,
                              theRuns_.front() - 0.5,
                              theRuns_.back() + 0.5);
    hitsByRun_ = book<TH1D>(
        "histByRun", "n. of hits by run number;;n. of hits", runRange, theRuns_.front() - 0.5, theRuns_.back() + 0.5);

    trackRatesByRun_ = book<TH1D>("trackRatesByRun",
                                  "rate of AlCaReco Tracks by run number;;n. of tracks/s",
                                  runRange,
                                  theRuns_.front() - 0.5,
                                  theRuns_.back() + 0.5);
    eventRatesByRun_ = book<TH1D>("eventRatesByRun",
                                  "rate of AlCaReco Events by run number;;n. of events/s",
                                  runRange,
                                  theRuns_.front() - 0.5,
                                  theRuns_.back() + 0.5);

    hitsinBPixByRun_ = book<TH1D>("histinBPixByRun",
                                  "n. of hits in BPix by run number;;n. of BPix hits",
                                  runRange,
                                  theRuns_.front() - 0.5,
                                  theRuns_.back() + 0.5);
    hitsinFPixByRun_ = book<TH1D>("histinFPixByRun",
                                  "n. of hits in FPix by run number;;n. of FPix hits",
                                  runRange,
                                  theRuns_.front() - 0.5,
                                  theRuns_.back() + 0.5);

    for (const auto &the_r : theRuns_) {
      if (conditionsMap_.find(the_r)->second.first != 0) {
        auto indexing = (the_r - theRuns_.front()) + 1;
        double runTime = timeMap_.find(the_r)->second;

        edm::LogPrint("DMRChecker") << "run:" << the_r << " | isPeak: " << std::setw(4)
                                    << conditionsMap_.find(the_r)->second.first
                                    << "| B-field: " << conditionsMap_.find(the_r)->second.second << " [T]"
                                    << "| events: " << setw(10) << runInfoMap_.find(the_r)->second.first
                                    << "(rate: " << setw(10) << (runInfoMap_.find(the_r)->second.first) / runTime
                                    << " ev/s)"
                                    << ", tracks " << setw(10) << runInfoMap_.find(the_r)->second.second
                                    << "(rate: " << setw(10) << (runInfoMap_.find(the_r)->second.second) / runTime
                                    << " trk/s)" << std::endl;

        // int the_bin = modeByRun_->GetXaxis()->FindBin(the_r);
        modeByRun_->SetBinContent(indexing, conditionsMap_.find(the_r)->second.first);
        modeByRun_->GetXaxis()->SetBinLabel(indexing, Form("%d", the_r));
        fieldByRun_->SetBinContent(indexing, conditionsMap_.find(the_r)->second.second);
        fieldByRun_->GetXaxis()->SetBinLabel(indexing, Form("%d", the_r));

        tracksByRun_->SetBinContent(indexing, runInfoMap_.find(the_r)->second.first);
        tracksByRun_->GetXaxis()->SetBinLabel(indexing, Form("%d", the_r));
        hitsByRun_->SetBinContent(indexing, runInfoMap_.find(the_r)->second.second);
        hitsByRun_->GetXaxis()->SetBinLabel(indexing, Form("%d", the_r));

        hitsinBPixByRun_->SetBinContent(indexing, (runHitsMap_.find(the_r)->second)[0]);
        hitsinBPixByRun_->GetXaxis()->SetBinLabel(indexing, Form("%d", the_r));
        hitsinFPixByRun_->SetBinContent(indexing, (runHitsMap_.find(the_r)->second)[1]);
        hitsinFPixByRun_->GetXaxis()->SetBinLabel(indexing, Form("%d", the_r));

        trackRatesByRun_->SetBinContent(indexing, (runInfoMap_.find(the_r)->second.second) / runTime);
        trackRatesByRun_->GetXaxis()->SetBinLabel(indexing, Form("%d", the_r));
        eventRatesByRun_->SetBinContent(indexing, (runInfoMap_.find(the_r)->second.first) / runTime);
        eventRatesByRun_->GetXaxis()->SetBinLabel(indexing, Form("%d", the_r));

        constexpr const char *subdets[]{"BPix", "FPix", "TIB", "TID", "TOB", "TEC"};

        edm::LogPrint("DMRChecker") << "*******************************" << std::endl;
        edm::LogPrint("DMRChecker") << "Hits by SubDetector" << std::endl;
        int si = 0;
        for (const auto &entry : runHitsMap_.find(the_r)->second) {
          edm::LogPrint("DMRChecker") << subdets[si] << " " << entry << std::endl;
          si++;
        }
        edm::LogPrint("DMRChecker") << "*******************************" << std::endl;
      }

      // modeByRun_->GetXaxis()->SetBinLabel(the_r-theRuns_[0]+1,(const char*)the_r);
    }

    if (DEBUG)
      edm::LogPrint("DMRChecker") << __FILE__ << "@" << __FUNCTION__ << " L-" << __LINE__ << std::endl;

    // DMRs

    TFileDirectory DMeanR = fs->mkdir("DMRs");

    DMRBPixX_ = DMeanR.make<TH1D>("DMRBPix-X", "DMR of BPix-X;mean of X-residuals;modules", 100., -200, 200);
    DMRBPixY_ = DMeanR.make<TH1D>("DMRBPix-Y", "DMR of BPix-Y;mean of Y-residuals;modules", 100., -200, 200);

    DMRFPixX_ = DMeanR.make<TH1D>("DMRFPix-X", "DMR of FPix-X;mean of X-residuals;modules", 100., -200, 200);
    DMRFPixY_ = DMeanR.make<TH1D>("DMRFPix-Y", "DMR of FPix-Y;mean of Y-residuals;modules", 100., -200, 200);

    DMRTIB_ = DMeanR.make<TH1D>("DMRTIB", "DMR of TIB;mean of X-residuals;modules", 100., -200, 200);
    DMRTOB_ = DMeanR.make<TH1D>("DMRTOB", "DMR of TOB;mean of X-residuals;modules", 100., -200, 200);

    DMRTID_ = DMeanR.make<TH1D>("DMRTID", "DMR of TID;mean of X-residuals;modules", 100., -200, 200);
    DMRTEC_ = DMeanR.make<TH1D>("DMRTEC", "DMR of TEC;mean of X-residuals;modules", 100., -200, 200);

    TFileDirectory DMeanRSplit = fs->mkdir("SplitDMRs");

    DMRBPixXSplit_ = bookSplitDMRHistograms(DMeanRSplit, "BPix", "X", true);
    DMRBPixYSplit_ = bookSplitDMRHistograms(DMeanRSplit, "BPix", "Y", true);

    DMRFPixXSplit_ = bookSplitDMRHistograms(DMeanRSplit, "FPix", "X", false);
    DMRFPixYSplit_ = bookSplitDMRHistograms(DMeanRSplit, "FPix", "Y", false);

    DMRTIBSplit_ = bookSplitDMRHistograms(DMeanRSplit, "TIB", "X", true);
    DMRTOBSplit_ = bookSplitDMRHistograms(DMeanRSplit, "TOB", "X", true);

    // DRnRs
    TFileDirectory DRnRs = fs->mkdir("DRnRs");

    DRnRBPixX_ = DRnRs.make<TH1D>("DRnRBPix-X", "DRnR of BPix-X;rms of normalized X-residuals;modules", 100., 0., 3.);
    DRnRBPixY_ = DRnRs.make<TH1D>("DRnRBPix-Y", "DRnR of BPix-Y;rms of normalized Y-residuals;modules", 100., 0., 3.);

    DRnRFPixX_ = DRnRs.make<TH1D>("DRnRFPix-X", "DRnR of FPix-X;rms of normalized X-residuals;modules", 100., 0., 3.);
    DRnRFPixY_ = DRnRs.make<TH1D>("DRnRFPix-Y", "DRnR of FPix-Y;rms of normalized Y-residuals;modules", 100., 0., 3.);

    DRnRTIB_ = DRnRs.make<TH1D>("DRnRTIB", "DRnR of TIB;rms of normalized X-residuals;modules", 100., 0., 3.);
    DRnRTOB_ = DRnRs.make<TH1D>("DRnRTOB", "DRnR of TOB;rms of normalized Y-residuals;modules", 100., 0., 3.);

    DRnRTID_ = DRnRs.make<TH1D>("DRnRTID", "DRnR of TID;rms of normalized X-residuals;modules", 100., 0., 3.);
    DRnRTEC_ = DRnRs.make<TH1D>("DRnRTEC", "DRnR of TEC;rms of normalized Y-residuals;modules", 100., 0., 3.);

    // initialize the topology first

    SiPixelPI::PhaseInfo ph_info(phase_);
    const char *path_toTopologyXML = ph_info.pathToTopoXML();
    const TrackerTopology standaloneTopo =
        StandaloneTrackerTopology::fromTrackerParametersXMLFile(edm::FileInPath(path_toTopologyXML).fullPath());

    // initialize PixelRegionContainers
    PixelDMRS_x_ByLayer = std::make_unique<PixelRegions::PixelRegionContainers>(&standaloneTopo, ph_info.phase());
    PixelDMRS_y_ByLayer = std::make_unique<PixelRegions::PixelRegionContainers>(&standaloneTopo, ph_info.phase());

    // book PixelRegionContainers
    PixelDMRS_x_ByLayer->bookAll("Barrel Pixel DMRs", "median(x'_{pred}-x'_{hit}) [#mum]", "# modules", 100, -50, 50);
    PixelDMRS_y_ByLayer->bookAll("Barrel Pixel DMRs", "median(y'_{pred}-y'_{hit}) [#mum]", "# modules", 100, -50, 50);

    if (DEBUG) {
      auto dets = PixelRegions::attachedDets(PixelRegions::PixelId::L1, &standaloneTopo, phase_);
      for (const auto &det : dets) {
        auto myLocalTopo = PixelDMRS_x_ByLayer->getTheTopo();
        edm::LogVerbatim("DMRChecker") << myLocalTopo->print(det) << std::endl;
      }
    }

    // pixel

    for (auto &bpixid : resDetailsBPixX_) {
      DMRBPixX_->Fill(bpixid.second.runningMeanOfRes_);
      if (phase_ == SiPixelPI::phase::one) {
        pixelmap->fillBarrelBin("DMRsX", bpixid.first, bpixid.second.runningMeanOfRes_);
        fullPixelmapXDMR->fillTrackerMap(bpixid.first, bpixid.second.runningMeanOfRes_);
      }

      if (DEBUG) {
        auto myLocalTopo = PixelDMRS_x_ByLayer->getTheTopo();
        edm::LogPrint("DMRChecker") << myLocalTopo->print(bpixid.first) << std::endl;
      }

      PixelDMRS_x_ByLayer->fill(bpixid.first, bpixid.second.runningMeanOfRes_);

      // split DMR
      if (bpixid.second.rDirection > 0) {
        DMRBPixXSplit_[0]->Fill(bpixid.second.runningMeanOfRes_);
      } else {
        DMRBPixXSplit_[1]->Fill(bpixid.second.runningMeanOfRes_);
      }

      if (bpixid.second.hitCount < 2)
        DRnRBPixX_->Fill(-1);
      else
        DRnRBPixX_->Fill(sqrt(bpixid.second.runningNormVarOfRes_ / (bpixid.second.hitCount - 1)));
    }

    for (auto &bpixid : resDetailsBPixY_) {
      DMRBPixY_->Fill(bpixid.second.runningMeanOfRes_);
      if (phase_ == SiPixelPI::phase::one) {
        pixelmap->fillBarrelBin("DMRsY", bpixid.first, bpixid.second.runningMeanOfRes_);
        fullPixelmapYDMR->fillTrackerMap(bpixid.first, bpixid.second.runningMeanOfRes_);
      }

      PixelDMRS_y_ByLayer->fill(bpixid.first, bpixid.second.runningMeanOfRes_);

      // split DMR
      if (bpixid.second.rDirection > 0) {
        DMRBPixYSplit_[0]->Fill(bpixid.second.runningMeanOfRes_);
      } else {
        DMRBPixYSplit_[1]->Fill(bpixid.second.runningMeanOfRes_);
      }

      if (bpixid.second.hitCount < 2)
        DRnRBPixY_->Fill(-1);
      else
        DRnRBPixY_->Fill(sqrt(bpixid.second.runningNormVarOfRes_ / (bpixid.second.hitCount - 1)));
    }

    for (auto &fpixid : resDetailsFPixX_) {
      DMRFPixX_->Fill(fpixid.second.runningMeanOfRes_);
      if (phase_ == SiPixelPI::phase::one) {
        pixelmap->fillForwardBin("DMRsX", fpixid.first, fpixid.second.runningMeanOfRes_);
        fullPixelmapXDMR->fillTrackerMap(fpixid.first, fpixid.second.runningMeanOfRes_);
      }
      PixelDMRS_x_ByLayer->fill(fpixid.first, fpixid.second.runningMeanOfRes_);

      // split DMR
      if (fpixid.second.zDirection > 0) {
        DMRFPixXSplit_[0]->Fill(fpixid.second.runningMeanOfRes_);
      } else {
        DMRFPixXSplit_[1]->Fill(fpixid.second.runningMeanOfRes_);
      }

      if (fpixid.second.hitCount < 2)
        DRnRFPixX_->Fill(-1);
      else
        DRnRFPixX_->Fill(sqrt(fpixid.second.runningNormVarOfRes_ / (fpixid.second.hitCount - 1)));
    }

    for (auto &fpixid : resDetailsFPixY_) {
      DMRFPixY_->Fill(fpixid.second.runningMeanOfRes_);
      if (phase_ == SiPixelPI::phase::one) {
        pixelmap->fillForwardBin("DMRsY", fpixid.first, fpixid.second.runningMeanOfRes_);
        fullPixelmapXDMR->fillTrackerMap(fpixid.first, fpixid.second.runningMeanOfRes_);
      }
      PixelDMRS_y_ByLayer->fill(fpixid.first, fpixid.second.runningMeanOfRes_);

      // split DMR
      if (fpixid.second.zDirection > 0) {
        DMRFPixYSplit_[0]->Fill(fpixid.second.runningMeanOfRes_);
      } else {
        DMRFPixYSplit_[1]->Fill(fpixid.second.runningMeanOfRes_);
      }

      if (fpixid.second.hitCount < 2)
        DRnRFPixY_->Fill(-1);
      else
        DRnRFPixY_->Fill(sqrt(fpixid.second.runningNormVarOfRes_ / (fpixid.second.hitCount - 1)));
    }

    // strips

    for (auto &tibid : resDetailsTIB_) {
      DMRTIB_->Fill(tibid.second.runningMeanOfRes_);

      // split DMR
      if (tibid.second.rDirection > 0) {
        DMRTIBSplit_[0]->Fill(tibid.second.runningMeanOfRes_);
      } else {
        DMRTIBSplit_[1]->Fill(tibid.second.runningMeanOfRes_);
      }

      if (tibid.second.hitCount < 2)
        DRnRTIB_->Fill(-1);
      else
        DRnRTIB_->Fill(sqrt(tibid.second.runningNormVarOfRes_ / (tibid.second.hitCount - 1)));
    }

    for (auto &tobid : resDetailsTOB_) {
      DMRTOB_->Fill(tobid.second.runningMeanOfRes_);

      // split DMR
      if (tobid.second.rDirection > 0) {
        DMRTOBSplit_[0]->Fill(tobid.second.runningMeanOfRes_);
      } else {
        DMRTOBSplit_[1]->Fill(tobid.second.runningMeanOfRes_);
      }

      if (tobid.second.hitCount < 2)
        DRnRTOB_->Fill(-1);
      else
        DRnRTOB_->Fill(sqrt(tobid.second.runningNormVarOfRes_ / (tobid.second.hitCount - 1)));
    }

    for (auto &tidid : resDetailsTID_) {
      DMRTID_->Fill(tidid.second.runningMeanOfRes_);

      if (tidid.second.hitCount < 2)
        DRnRTID_->Fill(-1);
      else
        DRnRTID_->Fill(sqrt(tidid.second.runningNormVarOfRes_ / (tidid.second.hitCount - 1)));
    }

    for (auto &tecid : resDetailsTEC_) {
      DMRTEC_->Fill(tecid.second.runningMeanOfRes_);

      if (tecid.second.hitCount < 2)
        DRnRTEC_->Fill(-1);
      else
        DRnRTEC_->Fill(sqrt(tecid.second.runningNormVarOfRes_ / (tecid.second.hitCount - 1)));
    }

    edm::LogPrint("DMRChecker") << "n. of bpix modules " << resDetailsBPixX_.size() << std::endl;
    edm::LogPrint("DMRChecker") << "n. of fpix modules " << resDetailsFPixX_.size() << std::endl;

    if (phase_ == SiPixelPI::phase::zero) {
      pmap->save(true, 0, 0, "PixelHitMap.pdf", 600, 800);
      pmap->save(true, 0, 0, "PixelHitMap.png", 500, 750);
    }

    tmap->save(true, 0, 0, "StripHitMap.pdf");
    tmap->save(true, 0, 0, "StripHitMap.png");

    if (phase_ == SiPixelPI::phase::one) {
      gStyle->SetPalette(kRainBow);
      pixelmap->beautifyAllHistograms();

      TCanvas cBX("CanvXBarrel", "CanvXBarrel", 1200, 1000);
      pixelmap->drawBarrelMaps("DMRsX", cBX);
      cBX.SaveAs("pixelBarrelDMR_x.png");

      TCanvas cFX("CanvXForward", "CanvXForward", 1600, 1000);
      pixelmap->drawForwardMaps("DMRsX", cFX);
      cFX.SaveAs("pixelForwardDMR_x.png");

      TCanvas cBY("CanvYBarrel", "CanvYBarrel", 1200, 1000);
      pixelmap->drawBarrelMaps("DMRsY", cBY);
      cBY.SaveAs("pixelBarrelDMR_y.png");

      TCanvas cFY("CanvXForward", "CanvXForward", 1600, 1000);
      pixelmap->drawForwardMaps("DMRsY", cFY);
      cFY.SaveAs("pixelForwardDMR_y.png");

      TCanvas cFullPixelxDMR("CanvFullPixelX", "CanvFullPixelX", 3000, 2000);
      fullPixelmapXDMR->printTrackerMap(cFullPixelxDMR);
      cFullPixelxDMR.SaveAs("fullPixelDMR_x.png");

      TCanvas cFullPixelyDMR("CanvFullPixelX", "CanvFullPixelY", 3000, 2000);
      fullPixelmapXDMR->printTrackerMap(cFullPixelyDMR);
      cFullPixelyDMR.SaveAs("fullPixelDMR_y.png");
    }

    // take care now of the 1D histograms
    gStyle->SetOptStat("emr");
    PixelDMRS_x_ByLayer->beautify(2, 0);
    PixelDMRS_y_ByLayer->beautify(2, 0);

    TCanvas DMRxBarrel("DMRxBarrelCanv", "x-coordinate", 1400, 1200);
    DMRxBarrel.Divide(2, 2);
    PixelDMRS_x_ByLayer->draw(DMRxBarrel, true, "HISTS");
    adjustCanvases(DMRxBarrel, true);
    for (unsigned int c = 1; c <= 4; c++) {
      DMRxBarrel.cd(c)->Update();
    }
    PixelDMRS_x_ByLayer->stats();

    TCanvas DMRxForward("DMRxForwardCanv", "x-coordinate", 1400, 1200);
    DMRxForward.Divide(4, 3);
    PixelDMRS_x_ByLayer->draw(DMRxForward, false, "HISTS");
    adjustCanvases(DMRxForward, false);
    for (unsigned int c = 1; c <= 12; c++) {
      DMRxForward.cd(c)->Update();
    }
    PixelDMRS_x_ByLayer->stats();

    DMRxBarrel.SaveAs("DMR_x_Barrel_ByLayer.png");
    DMRxForward.SaveAs("DMR_x_Forward_ByRing.png");

    TCanvas DMRyBarrel("DMRyBarrelCanv", "y-coordinate", 1400, 1200);
    DMRyBarrel.Divide(2, 2);
    PixelDMRS_y_ByLayer->draw(DMRyBarrel, true, "HISTS");
    adjustCanvases(DMRyBarrel, true);
    for (unsigned int c = 1; c <= 4; c++) {
      DMRyBarrel.cd(c)->Update();
    }
    PixelDMRS_y_ByLayer->stats();

    TCanvas DMRyForward("DMRyForwardCanv", "y-coordinate", 1400, 1200);
    DMRyForward.Divide(4, 3);
    PixelDMRS_y_ByLayer->draw(DMRyForward, false, "HISTS");
    adjustCanvases(DMRyForward, false);
    for (unsigned int c = 1; c <= 12; c++) {
      DMRyForward.cd(c)->Update();
    }
    PixelDMRS_y_ByLayer->stats();

    DMRyBarrel.SaveAs("DMR_y_Barrel_ByLayer.png");
    DMRyForward.SaveAs("DMR_y_Forward_ByRing.png");
  }

  //*************************************************************
  // Adjust canvas for DMRs
  //*************************************************************
  void adjustCanvases(TCanvas &canvas, bool isBarrel) {
    unsigned int maxPads = isBarrel ? 4 : 12;
    for (unsigned int c = 1; c <= maxPads; c++) {
      canvas.cd(c);
      SiPixelPI::adjustCanvasMargins(canvas.cd(c), 0.06, 0.12, 0.12, 0.05);
    }

    auto ltx = TLatex();
    ltx.SetTextFont(62);
    ltx.SetTextSize(0.05);
    ltx.SetTextAlign(11);

    std::string toAppend = canvas.GetTitle();

    for (unsigned int c = 1; c <= maxPads; c++) {
      auto index = isBarrel ? c - 1 : c + 3;
      canvas.cd(c);
      ltx.DrawLatexNDC(gPad->GetLeftMargin(),
                       1 - gPad->GetTopMargin() + 0.01,
                       (PixelRegions::IDlabels.at(index) + " " + toAppend).c_str());
    }
  }

  //*************************************************************
  // check if the hit is 2D
  //*************************************************************
  bool isHit2D(const TrackingRecHit &hit) {
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
            edm::LogError("UnknownType") << "@SUB=DMRChecker::isHit2D"
                                         << "Tracker hit not in pixel, neither SiStripRecHit[12]D nor "
                                         << "SiStripMatchedRecHit2D nor ProjectedSiStripRecHit2D.";
            return false;
          }
        }
      } else {  // not tracker??
        edm::LogWarning("DetectorMismatch") << "@SUB=DMRChecker::isHit2D"
                                            << "Hit not in tracker with 'official' dimension >=2.";
        return true;  // dimension() >= 2 so accept that...
      }
    }
    // never reached...
  }

  //*************************************************************
  // Generic booker of split DMRs
  //*************************************************************
  std::array<TH1D *, 2> bookSplitDMRHistograms(TFileDirectory dir,
                                               std::string subdet,
                                               std::string vartype,
                                               bool isBarrel) {
    TH1F::SetDefaultSumw2(kTRUE);

    std::array<TH1D *, 2> out;
    std::array<std::string, 2> sign_name = {{"plus", "minus"}};
    std::array<std::string, 2> sign = {{">0", "<0"}};
    for (unsigned int i = 0; i < 2; i++) {
      const char *name_;
      const char *title_;
      const char *axisTitle_;

      if (isBarrel) {
        name_ = Form("DMR%s_%s_rDir%s", subdet.c_str(), vartype.c_str(), sign_name[i].c_str());
        title_ = Form("Split DMR of %s-%s (rDir%s)", subdet.c_str(), vartype.c_str(), sign[i].c_str());
        axisTitle_ = Form("mean of %s-residuals (rDir%s);modules", vartype.c_str(), sign[i].c_str());
      } else {
        name_ = Form("DMR%s_%s_zDir%s", subdet.c_str(), vartype.c_str(), sign_name[i].c_str());
        title_ = Form("Split DMR of %s-%s (zDir%s)", subdet.c_str(), vartype.c_str(), sign[i].c_str());
        axisTitle_ = Form("mean of %s-residuals (zDir%s);modules", vartype.c_str(), sign[i].c_str());
      }

      out[i] = dir.make<TH1D>(name_, fmt::sprintf("%s;%s", title_, axisTitle_).c_str(), 100., -200, 200);
    }
    return out;
  }

  //*************************************************************
  // Generic booker function
  //*************************************************************
  std::map<unsigned int, TH1D *> bookResidualsHistogram(
      TFileDirectory dir, unsigned int theNLayers, std::string resType, std::string varType, std::string detType) {
    TH1F::SetDefaultSumw2(kTRUE);

    std::pair<double, double> limits;

    if (varType.find("Res") != std::string::npos) {
      limits = std::make_pair(-1000., 1000);
    } else {
      limits = std::make_pair(-3., 3.);
    }

    std::map<unsigned int, TH1D *> h;

    for (unsigned int i = 1; i <= theNLayers; i++) {
      const char *name_;
      const char *title_;
      std::string xAxisTitle_;

      if (varType.find("Res") != std::string::npos) {
        xAxisTitle_ = fmt::sprintf("res_{%s'} [#mum]", resType);
      } else {
        xAxisTitle_ = fmt::sprintf("res_{%s'}/#sigma_{res_{%s`}}", resType, resType);
      }

      unsigned int side = -1;
      if (detType.find("FPix") != std::string::npos) {
        side = (i - 1) / 3 + 1;
        unsigned int plane = (i - 1) % 3 + 1;

        std::string theSide = "";
        if (side == 1) {
          theSide = "Z-";
        } else {
          theSide = "Z+";
        }

        name_ = Form("h_%s%s%s_side%i_disk%i", detType.c_str(), varType.c_str(), resType.c_str(), side, plane);
        title_ = Form("%s (%s, disk %i) track %s-%s;%s;hits",
                      detType.c_str(),
                      theSide.c_str(),
                      plane,
                      resType.c_str(),
                      varType.c_str(),
                      xAxisTitle_.c_str());

      } else {
        name_ = Form("h_%s%s%s_layer%i", detType.c_str(), varType.c_str(), resType.c_str(), i);
        title_ = Form("%s (layer %i) track %s-%s;%s;hits",
                      detType.c_str(),
                      i,
                      resType.c_str(),
                      varType.c_str(),
                      xAxisTitle_.c_str());

        //edm::LogPrint("DMRChecker")<<"bookResidualsHistogram(): "<<i<<" layer:"<<i<<std::endl;
      }

      h[i] = dir.make<TH1D>(name_, title_, 100, limits.first, limits.second);
    }

    return h;
  }

  //*************************************************************
  // Generic filler function
  //*************************************************************
  void fillByIndex(std::map<unsigned int, TH1D *> &h, unsigned int index, double x) {
    if (h.count(index) != 0) {
      //if(TString(h[index]->GetName()).Contains("BPix"))
      //edm::LogPrint("DMRChecker")<<"fillByIndex() index: "<< index << " filling histogram: "<< h[index]->GetName() << std::endl;

      double min = h[index]->GetXaxis()->GetXmin();
      double max = h[index]->GetXaxis()->GetXmax();
      if (x < min)
        h[index]->Fill(min);
      else if (x >= max)
        h[index]->Fill(0.99999 * max);
      else
        h[index]->Fill(x);
    }
  }

  //*************************************************************
  // Implementation of the online variance algorithm
  // as in https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
  //*************************************************************
  void updateOnlineMomenta(running::estimatorMap &myDetails, uint32_t theID, float the_data, float the_pull) {
    myDetails[theID].hitCount += 1;

    float delta = 0;
    float n_delta = 0;

    if (myDetails[theID].hitCount != 1) {
      delta = the_data - myDetails[theID].runningMeanOfRes_;
      n_delta = the_pull - myDetails[theID].runningNormMeanOfRes_;
      myDetails[theID].runningMeanOfRes_ += (delta / myDetails[theID].hitCount);
      myDetails[theID].runningNormMeanOfRes_ += (n_delta / myDetails[theID].hitCount);
    } else {
      myDetails[theID].runningMeanOfRes_ = the_data;
      myDetails[theID].runningNormMeanOfRes_ = the_pull;
    }

    float delta2 = the_data - myDetails[theID].runningMeanOfRes_;
    float n_delta2 = the_pull - myDetails[theID].runningNormMeanOfRes_;

    myDetails[theID].runningVarOfRes_ += delta * delta2;
    myDetails[theID].runningNormVarOfRes_ += n_delta * n_delta2;
  }

  //*************************************************************
  // Fill the histograms using the running::estimatorMap
  //**************************************************************
  void fillDMRs(const running::estimatorMap &myDetails,
                TH1D *DMR,
                TH1D *DRnR,
                std::array<TH1D *, 2> DMRSplit,
                std::unique_ptr<PixelRegions::PixelRegionContainers> regionalDMR) {
    for (const auto &element : myDetails) {
      // DMR
      DMR->Fill(element.second.runningMeanOfRes_);

      // DMR by layer
      if (regionalDMR.get()) {
        regionalDMR->fill(element.first, element.second.runningMeanOfRes_);
      }

      // split DMR
      if (element.second.rOrZDirection > 0) {
        DMRSplit[0]->Fill(element.second.runningMeanOfRes_);
      } else {
        DMRSplit[1]->Fill(element.second.runningMeanOfRes_);
      }

      // DRnR
      if (element.second.hitCount < 2) {
        DRnR->Fill(-1);
      } else {
        DRnR->Fill(sqrt(element.second.runningNormVarOfRes_ / (element.second.hitCount - 1)));
      }
    }
  }
};

//*************************************************************
void DMRChecker::fillDescriptions(edm::ConfigurationDescriptions &descriptions)
//*************************************************************
{
  edm::ParameterSetDescription desc;
  desc.setComment("Generic track analyzer to check ALCARECO sample quantities / compute fast DMRs");
  desc.add<edm::InputTag>("TkTag", edm::InputTag("generalTracks"));
  desc.add<edm::InputTag>("TriggerResultsTag", edm::InputTag("TriggerResults", "", "HLT"));
  desc.add<edm::InputTag>("BeamSpotTag", edm::InputTag("offlineBeamSpot"));
  desc.add<edm::InputTag>("VerticesTag", edm::InputTag("offlinePrimaryVertices"));
  desc.add<bool>("isCosmics", false);
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(DMRChecker);

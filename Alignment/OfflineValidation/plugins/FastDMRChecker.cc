// -*- C++ -*-
//
// Package:    Alignment/OfflineValidation
// Class:      FastDMRChecker
//
/**\class FastDMRChecker FastDMRChecker.cc DMRChecker/DMRChecker/plugins/FastDMRChecker.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Marco Musich
//         Created:  Tue, 30 Nov 2021 11:08:17 GMT
//
//

// system include files
#include <cmath>
#include <memory>
#include <vector>

// user include files
#include "Alignment/OfflineValidation/interface/DMRHelper.h"
#include "Alignment/OfflineValidation/interface/TrackerValidationVariables.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CommonTools/Utils/interface/TFileDirectory.h"

// ROOT includes
#include "TH1F.h"

//
// class declaration
//

using reco::TrackCollection;
class FastDMRChecker : public edm::one::EDAnalyzer<edm::one::SharedResources, edm::one::WatchRuns> {
public:
  explicit FastDMRChecker(const edm::ParameterSet&);
  ~FastDMRChecker() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  // framework methods
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override;
  void endJob() override;

  // user defined methods
  void setOrientations(DMRHelper::EstimatorMap& myDetails, const uint32_t& theID, const TrackerGeometry& tkgeo);

  void fillDMRs(const DMRHelper::EstimatorMap& myDetails, TH1D* DMR, TH1D* DRnR, std::array<TH1D*, 2> DMRSplit);

  // lazily books (on first use) the by-layer / by-disk residual and pull
  // histogram for a given (subdet, layer) key, then delegates to the
  // shared DMRHelper::fillByLayer() to fill it.
  void fillByLayer(const std::pair<std::string, int32_t>& key, bool isX, float residual, float pull);

  // ----------member data ---------------------------

  // by-layer / by-disk residuals and pulls, booked lazily into
  // m_byLayerResidualsDir on first hit for a given subdetector-layer key
  DMRHelper::HistoSet m_SubdetLayerResiduals;
  std::unique_ptr<TFileDirectory> m_byLayerResidualsDir;

  edm::Service<TFileService> fs;
  TrackerValidationVariables avalidator_;
  const bool applyVertexCut_;
  const int minHitsPerModule_;

  // event tokens
  const edm::EDGetTokenT<reco::VertexCollection> offlinePVToken_;

  // event setup tokens
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopologyRunToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeometryToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopologyEventToken_;

  // Pixel
  DMRHelper::EstimatorMap resDetailsBPixX_;
  DMRHelper::EstimatorMap resDetailsBPixY_;
  DMRHelper::EstimatorMap resDetailsFPixX_;
  DMRHelper::EstimatorMap resDetailsFPixY_;

  // Strips
  DMRHelper::EstimatorMap resDetailsTIB_;
  DMRHelper::EstimatorMap resDetailsTOB_;
  DMRHelper::EstimatorMap resDetailsTID_;
  DMRHelper::EstimatorMap resDetailsTEC_;

  // Pixel
  TH1D* DMRBPixX_;
  TH1D* DMRBPixY_;
  TH1D* DMRFPixX_;
  TH1D* DMRFPixY_;
  TH1D* DRnRBPixX_;
  TH1D* DRnRBPixY_;
  TH1D* DRnRFPixX_;
  TH1D* DRnRFPixY_;

  // Strips
  TH1D* DMRTIB_;
  TH1D* DMRTOB_;
  TH1D* DMRTID_;
  TH1D* DMRTEC_;
  TH1D* DRnRTIB_;
  TH1D* DRnRTOB_;
  TH1D* DRnRTID_;
  TH1D* DRnRTEC_;

  // Split DMRs
  std::array<TH1D*, 2> DMRBPixXSplit_;
  std::array<TH1D*, 2> DMRBPixYSplit_;
  std::array<TH1D*, 2> DMRFPixXSplit_;
  std::array<TH1D*, 2> DMRFPixYSplit_;
  std::array<TH1D*, 2> DMRTIBSplit_;
  std::array<TH1D*, 2> DMRTOBSplit_;
  std::array<TH1D*, 2> DMRTIDSplit_;
  std::array<TH1D*, 2> DMRTECSplit_;
};

//
// constructors and destructor
//
FastDMRChecker::FastDMRChecker(const edm::ParameterSet& iConfig)
    : avalidator_(iConfig, consumesCollector()),
      applyVertexCut_(iConfig.getUntrackedParameter<bool>("VertexCut", true)),
      minHitsPerModule_(iConfig.getParameter<int>("minHitsPerModule")),
      offlinePVToken_(consumes<reco::VertexCollection>(iConfig.getParameter<std::string>("VertexCollection"))),
      trackerTopologyRunToken_{esConsumes<TrackerTopology, TrackerTopologyRcd, edm::Transition::BeginRun>()},
      trackerGeometryToken_{esConsumes<TrackerGeometry, TrackerDigiGeometryRecord, edm::Transition::BeginRun>()},
      trackerTopologyEventToken_{esConsumes<TrackerTopology, TrackerTopologyRcd>()} {
  usesResource(TFileService::kSharedResource);  // for thread-efficient usage of TFileService

  // ROOT histograms should always carry Sumw2 for correct error propagation;
  // set this once here rather than repeatedly at every booking call.
  TH1F::SetDefaultSumw2(kTRUE);
}

void FastDMRChecker::endRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {}  //endRun

void FastDMRChecker::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  const TrackerGeometry& TG = iSetup.getData(trackerGeometryToken_);
  // Collect list of modules from Tracker Geometry
  auto ids = TG.detIds();
  for (DetId id : ids) {
    auto ModuleID = id.rawId();

    switch (id.subdetId()) {
      case PixelSubdetector::PixelBarrel:
        this->setOrientations(resDetailsBPixX_, ModuleID, TG);
        this->setOrientations(resDetailsBPixY_, ModuleID, TG);
        break;
      case PixelSubdetector::PixelEndcap:
        this->setOrientations(resDetailsFPixX_, ModuleID, TG);
        this->setOrientations(resDetailsFPixY_, ModuleID, TG);
        break;
      case StripSubdetector::TIB:
        this->setOrientations(resDetailsTIB_, ModuleID, TG);
        break;
      case StripSubdetector::TOB:
        this->setOrientations(resDetailsTOB_, ModuleID, TG);
        break;
      case StripSubdetector::TID:
        this->setOrientations(resDetailsTID_, ModuleID, TG);
        break;
      case StripSubdetector::TEC:
        this->setOrientations(resDetailsTEC_, ModuleID, TG);
        break;
      default:
        throw cms::Exception("Inconsistent Data") << "Unknown Tracker subdetector: " << id.subdetId();
    }
  }
}  //beginRun

//
// member functions
//
// ------------ method called for each event  ------------
void FastDMRChecker::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  const TrackerTopology& tTopo = iSetup.getData(trackerTopologyEventToken_);

  edm::Handle<reco::VertexCollection> vertices;
  if (applyVertexCut_) {
    iEvent.getByToken(offlinePVToken_, vertices);
    if (!vertices.isValid() || vertices->empty())
      return;
  }

  auto vtracks = std::vector<TrackerValidationVariables::AVTrackStruct>();
  avalidator_.fillTrackQuantities(
      iEvent,
      iSetup,
      // tell the validator to only look at good tracks
      [&](const reco::Track& track) -> bool {
        return (!applyVertexCut_ ||
                (track.pt() > 0.75 && std::abs(track.dxy(vertices->at(0).position())) < 5 * track.dxyError()));
      },
      vtracks);

  // loop on the tracks
  for (auto& track : vtracks) {
    // loop on the rechits
    for (auto& it : track.hits) {
      uint32_t RawId = it.rawDetId;
      auto id = DetId(RawId);
      uint32_t subid = id.subdetId();
      const auto& resX = it.resXprime * DMRHelper::cmToUm;
      const auto& pullX = it.resXprime / it.resXprimeErr;

      // identify (subdetector, layer/disk) for the by-layer residual/pull histograms
      const auto& subdetAndLayer = DMRHelper::findSubdetAndLayer(RawId, &tTopo);

      auto isPixel = subid == PixelSubdetector::PixelBarrel || subid == PixelSubdetector::PixelEndcap;
      if (isPixel) {
        // y-residuals only for pixels
        const auto& resY = it.resYprime * DMRHelper::cmToUm;
        const auto& pullY = it.resYprime / it.resYprimeErr;

        if (subid == PixelSubdetector::PixelBarrel) {
          DMRHelper::updateOnlineMomenta(resDetailsBPixX_, RawId, resX, pullX);
          DMRHelper::updateOnlineMomenta(resDetailsBPixY_, RawId, resY, pullY);
        } else if (subid == PixelSubdetector::PixelEndcap) {
          DMRHelper::updateOnlineMomenta(resDetailsFPixX_, RawId, resX, pullX);
          DMRHelper::updateOnlineMomenta(resDetailsFPixY_, RawId, resY, pullY);
        }  // if FPix

        this->fillByLayer(subdetAndLayer, /*isX=*/true, resX, pullX);
        this->fillByLayer(subdetAndLayer, /*isX=*/false, resY, pullY);
      }  // if Pixel
      else {  // these are Strips
        if (subid == StripSubdetector::TIB) {
          DMRHelper::updateOnlineMomenta(resDetailsTIB_, RawId, resX, pullX);
        } else if (subid == StripSubdetector::TOB) {
          DMRHelper::updateOnlineMomenta(resDetailsTOB_, RawId, resX, pullX);
        } else if (subid == StripSubdetector::TID) {
          DMRHelper::updateOnlineMomenta(resDetailsTID_, RawId, resX, pullX);
        } else if (subid == StripSubdetector::TEC) {
          DMRHelper::updateOnlineMomenta(resDetailsTEC_, RawId, resX, pullX);
        }

        // strips only carry a single (X-prime) residual coordinate
        this->fillByLayer(subdetAndLayer, /*isX=*/true, resX, pullX);
      }
    }  // loop on hits
  }  // loop on tracks
}

void FastDMRChecker::endJob() {
  // DMRs
  TFileDirectory DMeanR = fs->mkdir("DMRs");
  DMRBPixX_ = DMeanR.make<TH1D>("DMRBPix-X", "DMR of BPix-X;mean of X-residuals;modules", 101, -50.5, 50.5);
  DMRBPixY_ = DMeanR.make<TH1D>("DMRBPix-Y", "DMR of BPix-Y;mean of Y-residuals;modules", 101, -50.5, 50.5);

  DMRFPixX_ = DMeanR.make<TH1D>("DMRFPix-X", "DMR of FPix-X;mean of X-residuals;modules", 101, -50.5, 50.5);
  DMRFPixY_ = DMeanR.make<TH1D>("DMRFPix-Y", "DMR of FPix-Y;mean of Y-residuals;modules", 101, -50.5, 50.5);

  DMRTIB_ = DMeanR.make<TH1D>("DMRTIB", "DMR of TIB;mean of X-residuals;modules", 101, -50.5, 50.5);
  DMRTOB_ = DMeanR.make<TH1D>("DMRTOB", "DMR of TOB;mean of X-residuals;modules", 101, -50.5, 50.5);

  DMRTID_ = DMeanR.make<TH1D>("DMRTID", "DMR of TID;mean of X-residuals;modules", 101, -50.5, 50.5);
  DMRTEC_ = DMeanR.make<TH1D>("DMRTEC", "DMR of TEC;mean of X-residuals;modules", 101, -50.5, 50.5);

  TFileDirectory DMeanRSplit = fs->mkdir("SplitDMRs");
  DMRBPixXSplit_ = DMRHelper::bookSplitDMRHistograms(DMeanRSplit, "BPix", "X", true);
  DMRBPixYSplit_ = DMRHelper::bookSplitDMRHistograms(DMeanRSplit, "BPix", "Y", true);

  DMRFPixXSplit_ = DMRHelper::bookSplitDMRHistograms(DMeanRSplit, "FPix", "X", false);
  DMRFPixYSplit_ = DMRHelper::bookSplitDMRHistograms(DMeanRSplit, "FPix", "Y", false);

  DMRTIBSplit_ = DMRHelper::bookSplitDMRHistograms(DMeanRSplit, "TIB", "X", true);
  DMRTOBSplit_ = DMRHelper::bookSplitDMRHistograms(DMeanRSplit, "TOB", "X", true);

  DMRTIDSplit_ = DMRHelper::bookSplitDMRHistograms(DMeanRSplit, "TID", "X", false);
  DMRTECSplit_ = DMRHelper::bookSplitDMRHistograms(DMeanRSplit, "TEC", "X", false);

  // DRnRs
  TFileDirectory DRnRs = fs->mkdir("DRnRs");

  DRnRBPixX_ = DRnRs.make<TH1D>("DRnRBPix-X", "DRnR of BPix-X;rms of normalized X-residuals;modules", 100, 0., 2.);
  DRnRBPixY_ = DRnRs.make<TH1D>("DRnRBPix-Y", "DRnR of BPix-Y;rms of normalized Y-residuals;modules", 100, 0., 2.);

  DRnRFPixX_ = DRnRs.make<TH1D>("DRnRFPix-X", "DRnR of FPix-X;rms of normalized X-residuals;modules", 100, 0., 2.);
  DRnRFPixY_ = DRnRs.make<TH1D>("DRnRFPix-Y", "DRnR of FPix-Y;rms of normalized Y-residuals;modules", 100, 0., 2.);

  DRnRTIB_ = DRnRs.make<TH1D>("DRnRTIB", "DRnR of TIB;rms of normalized X-residuals;modules", 100, 0., 2.);
  DRnRTOB_ = DRnRs.make<TH1D>("DRnRTOB", "DRnR of TOB;rms of normalized X-residuals;modules", 100, 0., 2.);

  DRnRTID_ = DRnRs.make<TH1D>("DRnRTID", "DRnR of TID;rms of normalized X-residuals;modules", 100, 0., 2.);
  DRnRTEC_ = DRnRs.make<TH1D>("DRnRTEC", "DRnR of TEC;rms of normalized X-residuals;modules", 100, 0., 2.);

  // fill the distributions
  this->fillDMRs(resDetailsBPixX_, DMRBPixX_, DRnRBPixX_, DMRBPixXSplit_);
  this->fillDMRs(resDetailsBPixY_, DMRBPixY_, DRnRBPixY_, DMRBPixYSplit_);

  this->fillDMRs(resDetailsFPixX_, DMRFPixX_, DRnRFPixX_, DMRFPixXSplit_);
  this->fillDMRs(resDetailsFPixY_, DMRFPixY_, DRnRFPixY_, DMRFPixYSplit_);

  this->fillDMRs(resDetailsTIB_, DMRTIB_, DRnRTIB_, DMRTIBSplit_);
  this->fillDMRs(resDetailsTOB_, DMRTOB_, DRnRTOB_, DMRTOBSplit_);

  this->fillDMRs(resDetailsTID_, DMRTID_, DRnRTID_, DMRTIDSplit_);
  this->fillDMRs(resDetailsTEC_, DMRTEC_, DRnRTEC_, DMRTECSplit_);
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void FastDMRChecker::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<int>("minHitsPerModule", 10);
  desc.add<std::string>("VertexCollection", "offlinePrimaryVertices");
  desc.addUntracked<bool>("VertexCut", false);

  // specific for TrackerValidationVariables
  TrackerValidationVariables::fillPSetDescription(desc);

  descriptions.addWithDefaultLabel(desc);
}

//*************************************************************
// Lazily book (into m_byLayerResidualsDir) and fill the by-layer
// residual/pull histograms for a given (subdet, layer) key, via
// the shared DMRHelper::fillByLayer().
//*************************************************************
void FastDMRChecker::fillByLayer(const std::pair<std::string, int32_t>& key, bool isX, float residual, float pull) {
  if (!m_byLayerResidualsDir) {
    m_byLayerResidualsDir = std::make_unique<TFileDirectory>(fs->mkdir("ByLayerResiduals"));
  }
  DMRHelper::fillByLayer(m_SubdetLayerResiduals, *m_byLayerResidualsDir, key, isX, residual, pull);
}

void FastDMRChecker::setOrientations(DMRHelper::EstimatorMap& myDetails,
                                     const uint32_t& theID,
                                     const TrackerGeometry& tkgeom) {
  // if the detid has never occcurred yet, set the local orientations
  if (myDetails.find(theID) == myDetails.end()) {
    const auto& id = DetId(theID);
    uint32_t subDetId = id.subdetId();

    //variables concerning the tracker geometry
    const Surface::PositionType& gPModule = tkgeom.idToDet(theID)->position();
    const Surface& surface = tkgeom.idToDet(theID)->surface();
    //global Orientation of local coordinate system of dets/detUnits
    LocalPoint lUDirection(1., 0., 0.), lVDirection(0., 1., 0.), lWDirection(0., 0., 1.);
    GlobalPoint gUDirection = surface.toGlobal(lUDirection), gVDirection = surface.toGlobal(lVDirection),
                gWDirection = surface.toGlobal(lWDirection);
    double dR(999.), dZ(999.);

    // assign the rOrZDirection
    if (subDetId == PixelSubdetector::PixelBarrel || subDetId == StripSubdetector::TIB ||
        subDetId == StripSubdetector::TOB) {
      dR = gWDirection.perp() - gPModule.perp();
      dZ = gVDirection.z() - gPModule.z();
      if (dR >= 0.)
        myDetails[theID].rOrZDirection = 1;
      else
        myDetails[theID].rOrZDirection = -1;
    } else if (subDetId == PixelSubdetector::PixelEndcap) {
      dR = gUDirection.perp() - gPModule.perp();
      dZ = gWDirection.z() - gPModule.z();
      if (dZ >= 0.)
        myDetails[theID].rOrZDirection = 1;
      else
        myDetails[theID].rOrZDirection = -1;
    } else if (subDetId == StripSubdetector::TID || subDetId == StripSubdetector::TEC) {
      dR = gVDirection.perp() - gPModule.perp();
      dZ = gWDirection.z() - gPModule.z();
      if (dR >= 0.)
        myDetails[theID].rOrZDirection = 1;
      else
        myDetails[theID].rOrZDirection = -1;
    }

    // assingn the r-direction (barrel)
    if (dR >= 0.)
      myDetails[theID].rDirection = 1;
    else
      myDetails[theID].rDirection = -1;

    // assign the z-direction (endcaps)
    if (dZ >= 0.)
      myDetails[theID].zDirection = 1;
    else
      myDetails[theID].zDirection = -1;
  }
}

//*************************************************************
// Fill the histograms using the DMRHelper::EstimatorMap
//**************************************************************
void FastDMRChecker::fillDMRs(const DMRHelper::EstimatorMap& myDetails,
                              TH1D* DMR,
                              TH1D* DRnR,
                              std::array<TH1D*, 2> DMRSplit) {
  // protections
  if (!DMR) {
    edm::LogWarning("FastDMRChecker") << "DMR histogram not available! Skipping";
    return;
  }
  if (!DRnR) {
    edm::LogWarning("FastDMRChecker") << "DRnR histogram not available! Skipping";
    return;
  }
  if (!DMRSplit[0] || !DMRSplit[1]) {
    edm::LogWarning("FastDMRChecker") << "Splot DMRs histograms not available! Skipping";
    return;
  }

  for (const auto& element : myDetails) {
    if (element.second.hitCount < minHitsPerModule_)
      continue;

    // DMR
    DMR->Fill(element.second.runningMeanOfRes_);

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

//define this as a plug-in
DEFINE_FWK_MODULE(FastDMRChecker);

//******************************************************************************
//
// Part of the refactorisation of of the E/gamma pixel matching pre-2017
// This refactorisation converts the monolithic  approach to a series of
// independent producer modules, with each modules performing  a specific
// job as recommended by the 2017 tracker framework
//
// This module is called a Producer even though its not an ED producer
// This was done to be consistant with other TrackingRegion producers
// in RecoTracker/TkTrackingRegions
//
// The module closely follows the other TrackingRegion producers
// in RecoTracker/TkTrackingRegions and is intended to become an EDProducer
// by TrackingRegionEDProducerT<TrackingRegionsFromSuperClustersProducer>

// This module c tracking regions from the superclusters. It mostly
// replicates the functionality of the SeedFilter class
// although unlike that class, it does not actually create seeds
//
// Author : Sam Harper (RAL), 2017
//
//*******************************************************************************

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"

#include "TrackingTools/TrajectoryState/interface/ftsFromVertexToPoint.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "CommonTools/Utils/interface/StringToEnumValue.h"

//stick this in common tools
#include "TEnum.h"
#include "TEnumConstant.h"
namespace {
  template <typename MyEnum>
  MyEnum strToEnum(std::string const& enumConstName) {
    TEnum* en = TEnum::GetEnum(typeid(MyEnum));
    if (en != nullptr) {
      if (TEnumConstant const* enc = en->GetConstant(enumConstName.c_str())) {
        return static_cast<MyEnum>(enc->GetValue());
      } else {
        throw cms::Exception("Configuration") << enumConstName << " is not a valid member of " << typeid(MyEnum).name();
      }
    }
    throw cms::Exception("LogicError") << typeid(MyEnum).name() << " not recognised by ROOT";
  }
  template <>
  RectangularEtaPhiTrackingRegion::UseMeasurementTracker strToEnum(std::string const& enumConstName) {
    using MyEnum = RectangularEtaPhiTrackingRegion::UseMeasurementTracker;
    if (enumConstName == "kNever")
      return MyEnum::kNever;
    else if (enumConstName == "kForSiStrips")
      return MyEnum::kForSiStrips;
    else if (enumConstName == "kAlways")
      return MyEnum::kAlways;
    else {
      throw cms::Exception("InvalidConfiguration")
          << enumConstName << " is not a valid member of " << typeid(MyEnum).name()
          << " (or strToEnum needs updating, this is a manual translation found at " << __FILE__ << " line " << __LINE__
          << ")";
    }
  }

}  // namespace
class TrackingRegionsFromSuperClustersProducer : public TrackingRegionProducer {
public:
  enum class Charge { NEG = -1, POS = +1 };

public:
  TrackingRegionsFromSuperClustersProducer(const edm::ParameterSet& cfg, edm::ConsumesCollector&& cc);

  ~TrackingRegionsFromSuperClustersProducer() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  std::vector<std::unique_ptr<TrackingRegion>> regions(const edm::Event& iEvent,
                                                       const edm::EventSetup& iSetup) const override;

private:
  GlobalPoint getVtxPos(const edm::Event& iEvent, double& deltaZVertex) const;

  std::unique_ptr<TrackingRegion> createTrackingRegion(const reco::SuperCluster& superCluster,
                                                       const GlobalPoint& vtxPos,
                                                       const double deltaZVertex,
                                                       const Charge charge,
                                                       const MeasurementTrackerEvent* measTrackerEvent,
                                                       const MagneticField& magField) const;

private:
  void validateConfigSettings() const;

private:
  //there are 3 modes in which to define the Z area of the tracking region
  //1) from first vertex in the passed vertices collection +/- originHalfLength in z (useZInVertex=true)
  //2) the beamspot +/- nrSigmaForBSDeltaZ* zSigma of beamspot (useZInBeamspot=true)
  //   the zSigma of the beamspot can have a minimum value specified
  //   we do this because a common error is that beamspot has too small of a value
  //3) defaultZ_ +/- originHalfLength  (if useZInVertex and useZInBeamspot are both false)
  double ptMin_;
  double originRadius_;
  double originHalfLength_;
  double deltaEtaRegion_;
  double deltaPhiRegion_;
  bool useZInVertex_;
  bool useZInBeamspot_;
  double nrSigmaForBSDeltaZ_;
  double defaultZ_;
  double minBSDeltaZ_;
  bool precise_;
  RectangularEtaPhiTrackingRegion::UseMeasurementTracker whereToUseMeasTracker_;

  edm::EDGetTokenT<reco::VertexCollection> verticesToken_;
  edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
  edm::EDGetTokenT<MeasurementTrackerEvent> measTrackerEventToken_;
  std::vector<edm::EDGetTokenT<std::vector<reco::SuperClusterRef>>> superClustersTokens_;

  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magFieldToken_;
};

namespace {
  template <typename T>
  edm::Handle<T> getHandle(const edm::Event& event, const edm::EDGetTokenT<T>& token) {
    edm::Handle<T> handle;
    event.getByToken(token, handle);
    return handle;
  }
}  // namespace

TrackingRegionsFromSuperClustersProducer::TrackingRegionsFromSuperClustersProducer(const edm::ParameterSet& cfg,
                                                                                   edm::ConsumesCollector&& iC)
    : magFieldToken_{iC.esConsumes()} {
  edm::ParameterSet regionPSet = cfg.getParameter<edm::ParameterSet>("RegionPSet");

  ptMin_ = regionPSet.getParameter<double>("ptMin");
  originRadius_ = regionPSet.getParameter<double>("originRadius");
  originHalfLength_ = regionPSet.getParameter<double>("originHalfLength");
  deltaPhiRegion_ = regionPSet.getParameter<double>("deltaPhiRegion");
  deltaEtaRegion_ = regionPSet.getParameter<double>("deltaEtaRegion");
  useZInVertex_ = regionPSet.getParameter<bool>("useZInVertex");
  useZInBeamspot_ = regionPSet.getParameter<bool>("useZInBeamspot");
  nrSigmaForBSDeltaZ_ = regionPSet.getParameter<double>("nrSigmaForBSDeltaZ");
  defaultZ_ = regionPSet.getParameter<double>("defaultZ");
  minBSDeltaZ_ = regionPSet.getParameter<double>("minBSDeltaZ");
  precise_ = regionPSet.getParameter<bool>("precise");
  whereToUseMeasTracker_ = strToEnum<RectangularEtaPhiTrackingRegion::UseMeasurementTracker>(
      regionPSet.getParameter<std::string>("whereToUseMeasTracker"));

  validateConfigSettings();

  auto verticesTag = regionPSet.getParameter<edm::InputTag>("vertices");
  auto beamSpotTag = regionPSet.getParameter<edm::InputTag>("beamSpot");
  auto superClustersTags = regionPSet.getParameter<std::vector<edm::InputTag>>("superClusters");
  auto measTrackerEventTag = regionPSet.getParameter<edm::InputTag>("measurementTrackerEvent");

  if (useZInVertex_) {
    verticesToken_ = iC.consumes(verticesTag);
  } else {
    beamSpotToken_ = iC.consumes(beamSpotTag);
  }
  if (whereToUseMeasTracker_ != RectangularEtaPhiTrackingRegion::UseMeasurementTracker::kNever) {
    measTrackerEventToken_ = iC.consumes(measTrackerEventTag);
  }
  for (const auto& tag : superClustersTags) {
    superClustersTokens_.emplace_back(iC.consumes(tag));
  }
}

void TrackingRegionsFromSuperClustersProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<double>("ptMin", 1.5);
  desc.add<double>("originRadius", 0.2);
  desc.add<double>("originHalfLength", 15.0)
      ->setComment("z range is +/- this value except when using the beamspot (useZInBeamspot=true)");
  desc.add<double>("deltaPhiRegion", 0.4);
  desc.add<double>("deltaEtaRegion", 0.1);
  desc.add<bool>("useZInVertex", false)
      ->setComment("use the leading vertex  position +/-orginHalfLength, mutually exclusive with useZInBeamspot");
  desc.add<bool>("useZInBeamspot", true)
      ->setComment(
          "use the beamspot  position +/- nrSigmaForBSDeltaZ* sigmaZ_{bs}, mutually exclusive with useZInVertex");
  desc.add<double>("nrSigmaForBSDeltaZ", 3.0)
      ->setComment("# of sigma to extend the z region when using the beamspot, only active if useZInBeamspot=true");
  desc.add<double>("minBSDeltaZ", 0.0)
      ->setComment("a minimum value of the beamspot sigma z to use, only active if useZInBeamspot=true");
  desc.add<double>("defaultZ", 0.)
      ->setComment("the default z position, only used if useZInVertex and useZInBeamspot are both false");
  desc.add<bool>("precise", true);
  desc.add<std::string>("whereToUseMeasTracker", "kNever");
  desc.add<edm::InputTag>("beamSpot", edm::InputTag("hltOnlineBeamSpot"))
      ->setComment("only used if useZInBeamspot is true");
  desc.add<edm::InputTag>("vertices", edm::InputTag())->setComment("only used if useZInVertex is true");
  desc.add<std::vector<edm::InputTag>>("superClusters",
                                       std::vector<edm::InputTag>{edm::InputTag{"hltEgammaSuperClustersToPixelMatch"}});
  desc.add<edm::InputTag>("measurementTrackerEvent", edm::InputTag());

  edm::ParameterSetDescription descRegion;
  descRegion.add<edm::ParameterSetDescription>("RegionPSet", desc);

  descriptions.add("trackingRegionsFromSuperClusters", descRegion);
}

std::vector<std::unique_ptr<TrackingRegion>> TrackingRegionsFromSuperClustersProducer::regions(
    const edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  std::vector<std::unique_ptr<TrackingRegion>> trackingRegions;

  double deltaZVertex = 0;
  GlobalPoint vtxPos = getVtxPos(iEvent, deltaZVertex);

  const MeasurementTrackerEvent* measTrackerEvent = nullptr;
  if (!measTrackerEventToken_.isUninitialized()) {
    measTrackerEvent = getHandle(iEvent, measTrackerEventToken_).product();
  }
  auto const& magField = iSetup.getData(magFieldToken_);

  for (auto& superClustersToken : superClustersTokens_) {
    auto superClustersHandle = getHandle(iEvent, superClustersToken);
    for (auto& superClusterRef : *superClustersHandle) {
      //do both charge hypothesises
      trackingRegions.emplace_back(
          createTrackingRegion(*superClusterRef, vtxPos, deltaZVertex, Charge::POS, measTrackerEvent, magField));
      trackingRegions.emplace_back(
          createTrackingRegion(*superClusterRef, vtxPos, deltaZVertex, Charge::NEG, measTrackerEvent, magField));
    }
  }
  return trackingRegions;
}

GlobalPoint TrackingRegionsFromSuperClustersProducer::getVtxPos(const edm::Event& iEvent, double& deltaZVertex) const {
  if (useZInVertex_) {
    auto verticesHandle = getHandle(iEvent, verticesToken_);
    //we throw if the vertices are not there but if no vertex is
    //recoed in the event, we default to 0,0,defaultZ as the vertex
    if (!verticesHandle->empty()) {
      deltaZVertex = originHalfLength_;
      const auto& pv = verticesHandle->front();
      return GlobalPoint(pv.x(), pv.y(), pv.z());
    } else {
      deltaZVertex = originHalfLength_;
      return GlobalPoint(0, 0, defaultZ_);
    }
  } else {
    auto beamSpotHandle = getHandle(iEvent, beamSpotToken_);
    const reco::BeamSpot::Point& bsPos = beamSpotHandle->position();

    if (useZInBeamspot_) {
      //as this is what has been done traditionally for e/gamma, others just use sigmaZ
      const double bsSigmaZ = std::sqrt(beamSpotHandle->sigmaZ() * beamSpotHandle->sigmaZ() +
                                        beamSpotHandle->sigmaZ0Error() * beamSpotHandle->sigmaZ0Error());
      const double sigmaZ = std::max(bsSigmaZ, minBSDeltaZ_);
      deltaZVertex = nrSigmaForBSDeltaZ_ * sigmaZ;

      return GlobalPoint(bsPos.x(), bsPos.y(), bsPos.z());
    } else {
      deltaZVertex = originHalfLength_;
      return GlobalPoint(bsPos.x(), bsPos.y(), defaultZ_);
    }
  }
}

std::unique_ptr<TrackingRegion> TrackingRegionsFromSuperClustersProducer::createTrackingRegion(
    const reco::SuperCluster& superCluster,
    const GlobalPoint& vtxPos,
    const double deltaZVertex,
    const Charge charge,
    const MeasurementTrackerEvent* measTrackerEvent,
    const MagneticField& magField) const {
  const GlobalPoint clusterPos(superCluster.position().x(), superCluster.position().y(), superCluster.position().z());
  const double energy = superCluster.energy();

  auto fts = trackingTools::ftsFromVertexToPoint(magField, clusterPos, vtxPos, energy, static_cast<int>(charge));
  return std::make_unique<RectangularEtaPhiTrackingRegion>(fts.momentum(),
                                                           vtxPos,
                                                           ptMin_,
                                                           originRadius_,
                                                           deltaZVertex,
                                                           deltaEtaRegion_,
                                                           deltaPhiRegion_,
                                                           whereToUseMeasTracker_,
                                                           precise_,
                                                           measTrackerEvent);
}

void TrackingRegionsFromSuperClustersProducer::validateConfigSettings() const {
  if (useZInVertex_ && useZInBeamspot_) {
    throw cms::Exception("InvalidConfiguration")
        << " when constructing TrackingRegionsFromSuperClustersProducer both useZInVertex (" << useZInVertex_
        << ") and useZInBeamspot (" << useZInBeamspot_ << ") can not be true as they are mutually exclusive options"
        << std::endl;
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducerFactory.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionEDProducerT.h"
DEFINE_EDM_PLUGIN(TrackingRegionProducerFactory,
                  TrackingRegionsFromSuperClustersProducer,
                  "TrackingRegionsFromSuperClustersProducer");
using TrackingRegionsFromSuperClustersEDProducer = TrackingRegionEDProducerT<TrackingRegionsFromSuperClustersProducer>;
DEFINE_FWK_MODULE(TrackingRegionsFromSuperClustersEDProducer);

#include "CalibFormats/SiStripObjects/interface/SiStripDetInfo.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/SiStripCluster/interface/SiStripApproximateCluster.h"
#include "DataFormats/SiStripCluster/interface/SiStripApproximateClusterCollection.h"
#include "DataFormats/SiStripCluster/interface/SiStripApproximateClusterCollectionV2.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiStripCommon/interface/ConstantsForHardwareSystems.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoTracker/PixelLowPtUtilities/interface/ClusterShapeHitFilter.h"
#include "RecoTracker/PixelLowPtUtilities/interface/SlidingPeakFinder.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"

#include <vector>
#include <memory>

class SiStripClusters2ApproxClusters : public edm::stream::EDProducer<> {
public:
  explicit SiStripClusters2ApproxClusters(const edm::ParameterSet& conf);
  void produce(edm::Event&, const edm::EventSetup&) override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  template <typename CollectionType>
  void fillCollection(CollectionType& result,
                      const edmNew::DetSetVector<SiStripCluster>& clusterCollection,
                      edm::Event& event,
                      const edm::EventSetup& iSetup);

  edm::InputTag inputClusters;
  edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster> > clusterToken;

  unsigned int maxNSat;
  static constexpr double subclusterWindow_ = .7;
  static constexpr double seedCutMIPs_ = .35;
  static constexpr double seedCutSN_ = 7.;
  static constexpr double subclusterCutMIPs_ = .45;
  static constexpr double subclusterCutSN_ = 12.;

  edm::InputTag beamSpot_;
  edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;

  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tkGeomToken_;

  edm::FileInPath fileInPath_;
  SiStripDetInfo detInfo_;

  std::string csfLabel_;
  unsigned int version;
  unsigned int collectionVersion;
  edm::ESGetToken<ClusterShapeHitFilter, CkfComponentsRecord> csfToken_;

  edm::ESGetToken<SiStripNoises, SiStripNoisesRcd> stripNoiseToken_;
  edm::ESHandle<SiStripNoises> theNoise_;
};

SiStripClusters2ApproxClusters::SiStripClusters2ApproxClusters(const edm::ParameterSet& conf) {
  inputClusters = conf.getParameter<edm::InputTag>("inputClusters");
  maxNSat = conf.getParameter<unsigned int>("maxSaturatedStrips");

  clusterToken = consumes<edmNew::DetSetVector<SiStripCluster> >(inputClusters);

  beamSpot_ = conf.getParameter<edm::InputTag>("beamSpot");
  beamSpotToken_ = consumes<reco::BeamSpot>(beamSpot_);

  tkGeomToken_ = esConsumes();

  fileInPath_ = edm::FileInPath(SiStripDetInfoFileReader::kDefaultFile);
  detInfo_ = SiStripDetInfoFileReader::read(fileInPath_.fullPath());

  csfLabel_ = conf.getParameter<std::string>("clusterShapeHitFilterLabel");
  csfToken_ = esConsumes(edm::ESInputTag("", csfLabel_));

  version = conf.getParameter<unsigned int>("version");
  collectionVersion = conf.getParameter<unsigned int>("collectionVersion");

  // Validate version parameter
  if (version != 1 && version != 2) {
    throw cms::Exception("InvalidParameter") << "Invalid version: " << version << ". Must be 1 or 2.";
  }

  // Validate collectionVersion parameter
  if (collectionVersion != 1 && collectionVersion != 2) {
    throw cms::Exception("InvalidParameter")
        << "Invalid collectionVersion: " << collectionVersion << ". Must be '1' or '2'.";
  }

  stripNoiseToken_ = esConsumes();

  if (collectionVersion == 1) {
    produces<SiStripApproximateClusterCollection>();
  } else if (collectionVersion == 2) {
    produces<SiStripApproximateClusterCollectionV2>();
  }
}

void SiStripClusters2ApproxClusters::produce(edm::Event& event, edm::EventSetup const& iSetup) {
  const auto& clusterCollection = event.get(clusterToken);

  if (collectionVersion == 1) {
    auto result = std::make_unique<SiStripApproximateClusterCollection>();
    result->reserve(clusterCollection.size(), clusterCollection.dataSize());

    fillCollection(*result, clusterCollection, event, iSetup);
    event.put(std::move(result));
  } else if (collectionVersion == 2) {
    auto result = std::make_unique<SiStripApproximateClusterCollectionV2>();
    result->reserve(clusterCollection.size(), clusterCollection.dataSize());

    fillCollection(*result, clusterCollection, event, iSetup);
    event.put(std::move(result));
  }
}

template <typename CollectionType>
void SiStripClusters2ApproxClusters::fillCollection(CollectionType& result,
                                                    const edmNew::DetSetVector<SiStripCluster>& clusterCollection,
                                                    edm::Event& event,
                                                    const edm::EventSetup& iSetup) {
  auto const beamSpotHandle = event.getHandle(beamSpotToken_);
  auto const& bs = beamSpotHandle.isValid() ? *beamSpotHandle : reco::BeamSpot();
  if (not beamSpotHandle.isValid()) {
    edm::LogError("SiStripClusters2ApproxClusters")
        << "didn't find a valid beamspot with label \"" << beamSpot_.encode() << "\" -> using (0,0,0)";
  }

  const auto& tkGeom = &iSetup.getData(tkGeomToken_);
  const auto& theFilter = &iSetup.getData(csfToken_);
  const auto& theNoise_ = &iSetup.getData(stripNoiseToken_);

  float previous_barycenter = SiStripApproximateCluster::barycenterOffset_;
  unsigned int offset_module_change = 0;

  for (const auto& detClusters : clusterCollection) {
    auto ff = result.beginDet(detClusters.id());

    unsigned int detId = detClusters.id();
    const GeomDet* det = tkGeom->idToDet(detId);
    double nApvs = detInfo_.getNumberOfApvsAndStripLength(detId).first;
    double stripLength = detInfo_.getNumberOfApvsAndStripLength(detId).second;
    double barycenter_ypos = 0.5 * stripLength;

    const StripGeomDetUnit* stripDet = dynamic_cast<const StripGeomDetUnit*>(det);
    float mip = 3.9 / (sistrip::MeVperADCStrip / stripDet->surface().bounds().thickness());

    for (const auto& cluster : detClusters) {
      const LocalPoint& lp = LocalPoint(((cluster.barycenter() * 10 / (sistrip::STRIPS_PER_APV * nApvs)) -
                                         ((stripDet->surface().bounds().width()) * 0.5f)),
                                        barycenter_ypos - (0.5f * stripLength),
                                        0.);
      const GlobalPoint& gpos = det->surface().toGlobal(lp);
      GlobalPoint beamspot(bs.position().x(), bs.position().y(), bs.position().z());
      const GlobalVector& gdir = gpos - beamspot;
      const LocalVector& ldir = det->toLocal(gdir);

      int hitStrips;
      float hitPredPos;
      bool usable = theFilter->getSizes(detId, cluster, lp, ldir, hitStrips, hitPredPos);
      // (almost) same logic as in StripSubClusterShapeTrajectoryFilter
      bool isTrivial = (std::abs(hitPredPos) < 2.f && hitStrips <= 2);

      if (!usable || isTrivial) {
        SiStripApproximateCluster approxCluster(
            cluster, maxNSat, hitPredPos, true, version, previous_barycenter, offset_module_change);
        ff.push_back(approxCluster);
        previous_barycenter = approxCluster.getBarycenter(previous_barycenter, offset_module_change);
      } else {
        bool peakFilter = false;
        SlidingPeakFinder pf(std::max<int>(2, std::ceil(std::abs(hitPredPos) + subclusterWindow_)));
        float mipnorm = mip / std::abs(ldir.z());
        PeakFinderTest test(mipnorm,
                            detId,
                            cluster.firstStrip(),
                            theNoise_,
                            seedCutMIPs_,
                            seedCutSN_,
                            subclusterCutMIPs_,
                            subclusterCutSN_);
        peakFilter = pf.apply(cluster.amplitudes(), test);

        SiStripApproximateCluster approxCluster(
            cluster, maxNSat, hitPredPos, peakFilter, version, previous_barycenter, offset_module_change);
        ff.push_back(approxCluster);
        previous_barycenter = approxCluster.getBarycenter(previous_barycenter, offset_module_change);
      }
      offset_module_change = 0;
    }
    offset_module_change = nApvs * sistrip::STRIPS_PER_APV;
  }
}

void SiStripClusters2ApproxClusters::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("inputClusters", edm::InputTag("siStripClusters"));
  desc.add<unsigned int>("maxSaturatedStrips", 3);
  desc.add<std::string>("clusterShapeHitFilterLabel", "ClusterShapeHitFilter");  // add CSF label
  desc.add<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpot"));         // add BeamSpot tag
  desc.add<unsigned int>("version", 1);  // RawPrime version (1= default, 2= new v2 format)
  desc.add<unsigned int>(
      "collectionVersion",
      1);  // Collection version (1 for SiStripApproximateClusterCollection, 2 for SiStripApproximateClusterCollectionV2)
  descriptions.add("SiStripClusters2ApproxClusters", desc);
}

DEFINE_FWK_MODULE(SiStripClusters2ApproxClusters);

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

#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include "RecoLocalTracker/Records/interface/TkStripCPERecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/DNNStripCluter.h"
#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"

#include <vector>
#include <memory>

using namespace cms::Ort;

class SiStripClusters2ApproxClusters : public edm::stream::EDProducer<> {
public:
  explicit SiStripClusters2ApproxClusters(const edm::ParameterSet& conf);
  void produce(edm::Event&, const edm::EventSetup&) override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::InputTag inputClusters;
  edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster> > clusterToken;
  edm::EDGetTokenT<reco::TrackCollection> hlttracksToken_;

  unsigned int maxNSat;
  static constexpr double subclusterWindow_ = .7;
  static constexpr double seedCutMIPs_ = .35;
  static constexpr double seedCutSN_ = 7.;
  static constexpr double subclusterCutMIPs_ = .45;
  static constexpr double subclusterCutSN_ = 12.;

  edm::InputTag beamSpot_;
  edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;

  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tkGeomToken_;
  edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> ttbToken_;
  const TransientTrackBuilder* theTTrackBuilder;

  edm::FileInPath fileInPath_;
  SiStripDetInfo detInfo_;

  std::string csfLabel_;
  edm::ESGetToken<ClusterShapeHitFilter, CkfComponentsRecord> csfToken_;

  edm::ESGetToken<SiStripNoises, SiStripNoisesRcd> stripNoiseToken_;
  edm::ESHandle<SiStripNoises> theNoise_;

  edm::ESGetToken<StripClusterParameterEstimator, TkStripCPERecord> stripCPEToken_;
  edm::ESGetToken<Propagator, TrackingComponentsRecord> propagatorToken_;
  std::unique_ptr<cms::Ort::ONNXRuntime> ort_mSession = nullptr;
  float cutvalue;
  std::vector<std::string> input_training_vars;
};

SiStripClusters2ApproxClusters::SiStripClusters2ApproxClusters(const edm::ParameterSet& conf) {
  inputClusters = conf.getParameter<edm::InputTag>("inputClusters");
  maxNSat = conf.getParameter<unsigned int>("maxSaturatedStrips");

  clusterToken = consumes<edmNew::DetSetVector<SiStripCluster> >(inputClusters);
  hlttracksToken_ = consumes<reco::TrackCollection>(conf.getParameter<edm::InputTag>("hlttracks"));

  beamSpot_ = conf.getParameter<edm::InputTag>("beamSpot");
  beamSpotToken_ = consumes<reco::BeamSpot>(beamSpot_);

  tkGeomToken_ = esConsumes();
  ttbToken_ = esConsumes(edm::ESInputTag("", "TransientTrackBuilder"));

  fileInPath_ = edm::FileInPath(SiStripDetInfoFileReader::kDefaultFile);
  detInfo_ = SiStripDetInfoFileReader::read(fileInPath_.fullPath());

  csfLabel_ = conf.getParameter<std::string>("clusterShapeHitFilterLabel");
  csfToken_ = esConsumes(edm::ESInputTag("", csfLabel_));

  stripNoiseToken_ = esConsumes();
  stripCPEToken_         = esConsumes<StripClusterParameterEstimator, TkStripCPERecord>(edm::ESInputTag("", "StripCPEfromTrackAngle"));
  propagatorToken_ = esConsumes(edm::ESInputTag("", "PropagatorWithMaterialParabolicMf"));
  input_training_vars = {"avg_charge", "max_adc", "std_x","std_y","std_z", "max_adc_x", "max_adc_y", "max_adc_z", "diff_adc_pone",
		  "diff_adc_ptwo",
		  "diff_adc_pthree",
		  "diff_adc_mone",
		  "diff_adc_mtwo", "diff_adc_mthree",
		  "dr_min_pixelTrk"};
  cutvalue = 0.0780;
  auto session_options = ONNXRuntime::defaultSessionOptions(Backend::cpu);
  auto uOrtSession = std::make_unique<ONNXRuntime>("nn_model.onnx", &session_options);
  ort_mSession = std::move(uOrtSession);
  produces<SiStripApproximateClusterCollection>();
}

void SiStripClusters2ApproxClusters::produce(edm::Event& event, edm::EventSetup const& iSetup) {
  const auto& clusterCollection = event.get(clusterToken);
  const auto& hlttracks         = event.get(hlttracksToken_);
  const auto* stripCPE          = &iSetup.getData(stripCPEToken_);
  const Propagator* thePropagator = &iSetup.getData(propagatorToken_);
  theTTrackBuilder = &iSetup.getData(ttbToken_);
  auto result = std::make_unique<SiStripApproximateClusterCollection>();
  result->reserve(clusterCollection.size(), clusterCollection.dataSize());

  auto const beamSpotHandle = event.getHandle(beamSpotToken_);
  auto const& bs = beamSpotHandle.isValid() ? *beamSpotHandle : reco::BeamSpot();
  if (not beamSpotHandle.isValid()) {
    edm::LogError("SiStripClusters2ApproxClusters")
        << "didn't find a valid beamspot with label \"" << beamSpot_.encode() << "\" -> using (0,0,0)";
  }

  const auto& tkGeom = &iSetup.getData(tkGeomToken_);
  const auto& theFilter = &iSetup.getData(csfToken_);
  const auto& theNoise_ = &iSetup.getData(stripNoiseToken_);
  const auto tkDets = tkGeom->dets();

  for (const auto& detClusters : clusterCollection) {
    auto ff = result->beginDet(detClusters.id());

    unsigned int detId = detClusters.id();
    const auto& _detId = detId; // for the capture clause in the lambda function
    auto _det = std::find_if(tkDets.begin(), tkDets.end(), [_detId](auto& elem) -> bool {
       return (elem->geographicalId().rawId() == _detId);
    });
    const StripTopology& p = dynamic_cast<const StripGeomDetUnit*>(*_det)->specificTopology();
    const GeomDet* det = tkGeom->idToDet(detId);
    double nApvs = detInfo_.getNumberOfApvsAndStripLength(detId).first;
    double stripLength = detInfo_.getNumberOfApvsAndStripLength(detId).second;
    double barycenter_ypos = 0.5 * stripLength;

    const StripGeomDetUnit* stripDet = dynamic_cast<const StripGeomDetUnit*>(det);
    float mip = 3.9 / (sistrip::MeVperADCStrip / stripDet->surface().bounds().thickness());

    const GeomDetUnit* geomDet = tkGeom->idToDetUnit(detId);

    for (const auto& cluster : detClusters) {
      float dr_min_pixelTrk = 99;
      for ( const auto & trk : hlttracks) {
	 for (auto const& hit : trk.recHits()) {
            if (!hit->isValid()) continue;
	    if (hit->geographicalId() != detId) continue;
	    reco::TransientTrack tkTT = theTTrackBuilder->build(trk);
	    TrajectoryStateOnSurface tsos = thePropagator->propagate(tkTT.innermostMeasurementState(), geomDet->surface());
	    if (!tsos.isValid()) continue;
	    LocalPoint clusterLocal = stripCPE->localParameters(cluster, *stripDet, tsos).first;
	    LocalPoint trackLocal = geomDet->surface().toLocal(tsos.globalPosition());
            float dr = abs(trackLocal.x() - clusterLocal.x());
	    if (dr < dr_min_pixelTrk) dr_min_pixelTrk = dr;
	 }
      }
      std::vector<std::vector<float>> dnn_input_values = DNNStripCluster(cluster, tkGeom, detId, p, dr_min_pixelTrk, input_training_vars);
      const std::vector<std::string> inputname {"input"};
      std::vector<std::vector<float>> dnn_output = ort_mSession->run(inputname, dnn_input_values);
      //std::cout << "dnn value: " << dnn_output[0][0] << std::endl;
      if (dnn_output[0][0] < cutvalue) continue;
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
        ff.push_back(SiStripApproximateCluster(cluster, maxNSat, hitPredPos, true));
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

        ff.push_back(SiStripApproximateCluster(cluster, maxNSat, hitPredPos, peakFilter));
      }
    }
  }

  event.put(std::move(result));
}

void SiStripClusters2ApproxClusters::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("inputClusters", edm::InputTag("siStripClusters"));
  desc.add<edm::InputTag>("hlttracks", edm::InputTag("hltMergedTracksPPOnAA"));
  desc.add<unsigned int>("maxSaturatedStrips", 3);
  desc.add<std::string>("clusterShapeHitFilterLabel", "ClusterShapeHitFilter");  // add CSF label
  desc.add<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpot"));         // add BeamSpot tag
  descriptions.add("SiStripClusters2ApproxClusters", desc);
}

DEFINE_FWK_MODULE(SiStripClusters2ApproxClusters);

#ifndef RecoHGCal_TICL_TracksterLinkingAlgoBySkeletons_H
#define RecoHGCal_TICL_TracksterLinkingAlgoBySkeletons_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"
#include "RecoHGCal/TICL/interface/TracksterLinkingAlgoBase.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include <array>

namespace ticl {

  class TracksterLinkingbySkeletons : public TracksterLinkingAlgoBase {
  public:
    TracksterLinkingbySkeletons(const edm::ParameterSet& conf,
                                edm::ConsumesCollector iC,
                                cms::Ort::ONNXRuntime const* onnxRuntime = nullptr);

    ~TracksterLinkingbySkeletons() override {}

    void linkTracksters(const Inputs& input,
                        std::vector<Trackster>& resultTracksters,
                        std::vector<std::vector<unsigned int>>& linkedResultTracksters,
                        std::vector<std::vector<unsigned int>>& linkedTracksterIdToInputTracksterId) override;

    std::array<ticl::Vector, 3> findSkeletonNodes(const ticl::Trackster& trackster,
                                                  float lower_percentage,
                                                  float upper_percentage,
                                                  const std::vector<reco::CaloCluster>& layerClusters,
                                                  const hgcal::RecHitTools& rhtools);

    bool areCompatible(const ticl::Trackster& myTrackster,
                       const ticl::Trackster& otherTrackster,
                       const std::array<ticl::Vector, 3>& mySkeleton,
                       const std::array<ticl::Vector, 3>& otherSkeleton);

    void initialize(const HGCalDDDConstants* hgcons,
                    const hgcal::RecHitTools rhtools,
                    const edm::ESHandle<MagneticField> bfieldH,
                    const edm::ESHandle<Propagator> propH) override;

    static void fillPSetDescription(edm::ParameterSetDescription& iDesc) {
      iDesc.add<double>("track_time_quality_threshold", 0.5);
      iDesc.add<double>("wind", 0.036);
      iDesc.add<unsigned int>("min_num_lcs", 7);
      iDesc.add<double>("min_trackster_energy", 10.);
      iDesc.add<double>("pca_quality_th", 0.85);
      iDesc.add<double>("dot_prod_th", 0.97);
      iDesc.add<std::vector<double>>("max_distance_projective_sqr", {60., 60.});
      iDesc.add<std::vector<double>>("min_distance_z", {30., 30.});
      iDesc.add<std::vector<double>>("max_distance_projective_sqr_closest_points", {60., 60.});
      iDesc.add<std::vector<double>>("max_z_distance_closest_points", {35., 35.});
      iDesc.add<std::vector<double>>("cylinder_radius_sqr", {9., 9.});
      TracksterLinkingAlgoBase::fillPSetDescription(iDesc);
    }

  private:
    using Vector = ticl::Trackster::Vector;

    void buildLayers();

    void dumpLinksFound(std::vector<std::vector<unsigned>>& resultCollection, const char* label) const;

    float timing_quality_threshold_;
    float del_;
    unsigned int min_num_lcs_;
    float min_trackster_energy_;
    float pca_quality_th_;
    float dot_prod_th_;
    std::vector<double> max_distance_projective_sqr_;
    std::vector<double> min_distance_z_;
    std::vector<double> max_distance_projective_sqr_closest_points_;
    std::vector<double> max_z_distance_closest_points_;
    std::vector<double> cylinder_radius_sqr_;

    const HGCalDDDConstants* hgcons_;

    std::unique_ptr<GeomDet> firstDisk_[2];
    std::unique_ptr<GeomDet> interfaceDisk_[2];

    hgcal::RecHitTools rhtools_;

    edm::ESHandle<MagneticField> bfield_;
    edm::ESHandle<Propagator> propagator_;
  };

}  // namespace ticl

#endif

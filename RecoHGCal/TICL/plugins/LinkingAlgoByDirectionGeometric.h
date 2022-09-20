#ifndef RecoHGCal_TICL_LinkingAlgoByDirectionGeometric_H__
#define RecoHGCal_TICL_LinkingAlgoByDirectionGeometric_H__

#include <memory>
#include <array>
#include "RecoHGCal/TICL/plugins/LinkingAlgoBase.h"
#include "RecoHGCal/TICL/interface/commons.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
#include "DataFormats/HGCalReco/interface/TICLLayerTile.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

namespace ticl {
  class LinkingAlgoByDirectionGeometric final : public LinkingAlgoBase {
  public:
    LinkingAlgoByDirectionGeometric(const edm::ParameterSet &conf);
    ~LinkingAlgoByDirectionGeometric() override;

    void initialize(const HGCalDDDConstants *hgcons,
                    const hgcal::RecHitTools rhtools,
                    const edm::ESHandle<MagneticField> bfieldH,
                    const edm::ESHandle<Propagator> propH) override;

    void linkTracksters(const edm::Handle<std::vector<reco::Track>>,
                        const edm::ValueMap<float> &,
                        const edm::ValueMap<float> &,
                        const edm::ValueMap<float> &,
                        const std::vector<reco::Muon> &,
                        const edm::Handle<std::vector<Trackster>>,
                        std::vector<TICLCandidate> &,
                        std::vector<TICLCandidate> &) override;

    static void fillPSetDescription(edm::ParameterSetDescription &desc);

  private:
    typedef math::XYZVector Vector;

    void buildLayers();

    math::XYZVector propagateTrackster(const Trackster &t,
                                       const unsigned idx,
                                       float zVal,
                                       std::array<TICLLayerTile, 2> &tracksterTiles);

    void findTrackstersInWindow(const std::vector<std::pair<Vector, unsigned>> &seedingCollection,
                                const std::array<TICLLayerTile, 2> &tracksterTiles,
                                const std::vector<Vector> &tracksterPropPoints,
                                float delta,
                                unsigned trackstersSize,
                                std::vector<std::vector<unsigned>> &resultCollection,
                                bool useMask);

    bool timeAndEnergyCompatible(float &total_raw_energy,
                                 const reco::Track &track,
                                 const Trackster &trackster,
                                 const float &tkTime,
                                 const float &tkTimeErr,
                                 const float &tkTimeQual);

    void recordTrackster(const unsigned ts,  // trackster index
                         const std::vector<Trackster> &tracksters,
                         const edm::Handle<std::vector<Trackster>> tsH,
                         std::vector<unsigned> &ts_mask,
                         float &energy_in_candidate,
                         TICLCandidate &candidate);

    void dumpLinksFound(std::vector<std::vector<unsigned>> &resultCollection, const char *label) const;

    const float tkEnergyCut_ = 2.0f;
    const float maxDeltaT_ = 3.0f;
    const float del_tk_ts_layer1_;
    const float del_tk_ts_int_;
    const float del_ts_em_had_;
    const float del_ts_had_had_;

    const float timing_quality_threshold_;

    const StringCutObjectSelector<reco::Track> cutTk_;
    std::once_flag initializeGeometry_;

    const HGCalDDDConstants *hgcons_;

    std::unique_ptr<GeomDet> firstDisk_[2];
    std::unique_ptr<GeomDet> interfaceDisk_[2];

    hgcal::RecHitTools rhtools_;

    edm::ESHandle<MagneticField> bfield_;
    edm::ESHandle<Propagator> propagator_;
  };
}  // namespace ticl
#endif

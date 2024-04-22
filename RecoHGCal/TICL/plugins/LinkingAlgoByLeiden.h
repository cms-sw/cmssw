#ifndef RecoHGCal_TICL_LinkingAlgoByLeiden_H__
#define RecoHGCal_TICL_LinkingAlgoByLeiden_H__

#include <memory>
#include <array>
#include "RecoHGCal/TICL/plugins/LinkingAlgoBase.h"
#include "RecoHGCal/TICL/interface/commons.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/HGCalReco/interface/TICLGraph.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
#include "DataFormats/HGCalReco/interface/TICLLayerTile.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

namespace ticl {
  class LinkingAlgoByLeiden final : public LinkingAlgoBase {
  public:
    LinkingAlgoByLeiden(const edm::ParameterSet &conf);
    ~LinkingAlgoByLeiden() override;

    void initialize(const HGCalDDDConstants *hgcons,
                    const hgcal::RecHitTools rhtools,
                    const edm::ESHandle<MagneticField> bfieldH,
                    const edm::ESHandle<Propagator> propH) override;

    void linkTracksters(const edm::Handle<std::vector<reco::Track>>,
                        const edm::Handle<edm::ValueMap<float>>,
                        const edm::Handle<edm::ValueMap<float>>,
                        const edm::Handle<edm::ValueMap<float>>,
                        const std::vector<reco::Muon> &,
                        const edm::Handle<std::vector<Trackster>>,
                        const edm::Handle<TICLGraph> &,
                        const bool useMTDTiming,
                        std::vector<TICLCandidate> &,
                        std::vector<TICLCandidate> &) override;
    static void fillPSetDescription(edm::ParameterSetDescription &desc);

  private:
    void buildLayers();

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

auto moveNodesFast(TICLGraph const &graph, Partition const &partition);
#endif

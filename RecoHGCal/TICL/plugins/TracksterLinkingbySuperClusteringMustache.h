/*
TICL plugin for electron superclustering in HGCAL with Mustache algorithm
Authors : Theo Cuisset <theo.cuisset@cern.ch>, Shamik Ghosh <shamik.ghosh@cern.ch>
Date : 06/2024
*/

#ifndef RecoHGCal_TICL_TracksterLinkingSuperClusteringMustache_H
#define RecoHGCal_TICL_TracksterLinkingSuperClusteringMustache_H

#include <vector>

namespace cms {
  namespace Ort {
    class ONNXRuntime;
  }
}  // namespace cms

#include "RecoHGCal/TICL/interface/TracksterLinkingAlgoBase.h"
#include "RecoHGCal/TICL/interface/SuperclusteringDNNInputs.h"

#include "CondFormats/EcalObjects/interface/EcalMustacheSCParameters.h"
#include "CondFormats/DataRecord/interface/EcalMustacheSCParametersRcd.h"
#include "CondFormats/EcalObjects/interface/EcalSCDynamicDPhiParameters.h"
#include "CondFormats/DataRecord/interface/EcalSCDynamicDPhiParametersRcd.h"

namespace ticl {
  class Trackster;

  class TracksterLinkingbySuperClusteringMustache : public TracksterLinkingAlgoBase {
  public:
    TracksterLinkingbySuperClusteringMustache(const edm::ParameterSet& ps,
                                              edm::ConsumesCollector iC,
                                              cms::Ort::ONNXRuntime const* onnxRuntime = nullptr);
    /* virtual */ ~TracksterLinkingbySuperClusteringMustache() override {}
    static void fillPSetDescription(edm::ParameterSetDescription& iDesc);

    void linkTracksters(const Inputs& input,
                        std::vector<Trackster>& resultTracksters,
                        std::vector<std::vector<unsigned int>>& linkedResultTracksters,
                        std::vector<std::vector<unsigned int>>& linkedTracksterIdToInputTracksterId) override;
    void initialize(const HGCalDDDConstants* hgcons,
                    const hgcal::RecHitTools rhtools,
                    const edm::ESHandle<MagneticField> bfieldH,
                    const edm::ESHandle<Propagator> propH) override;

    virtual void setEvent(edm::Event& iEvent, edm::EventSetup const& iEventSetup) override;

  private:
    bool trackstersPassesPIDCut(const Trackster& ts) const;

    edm::ESGetToken<EcalMustacheSCParameters, EcalMustacheSCParametersRcd> ecalMustacheSCParametersToken_;
    edm::ESGetToken<EcalSCDynamicDPhiParameters, EcalSCDynamicDPhiParametersRcd> ecalSCDynamicDPhiParametersToken_;
    const EcalMustacheSCParameters* mustacheSCParams_;
    const EcalSCDynamicDPhiParameters* scDynamicDPhiParams_;

    float seedThresholdPt_;
    float candidateEnergyThreshold_;
    bool filterByTracksterPID_;
    std::vector<int> tracksterPIDCategoriesToFilter_;
    float PIDThreshold_;
  };

}  // namespace ticl

#endif

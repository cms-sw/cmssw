
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "L1Trigger/L1THGCal/interface/backend/HGCalTriggerClusterInterpreterBase.h"
#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerTools.h"

class HGCalTriggerClusterInterpretationEM : public HGCalTriggerClusterInterpreterBase {
public:
  HGCalTriggerClusterInterpretationEM();
  ~HGCalTriggerClusterInterpretationEM() override{};
  void initialize(const edm::ParameterSet& conf) final;
  void eventSetup(const edm::EventSetup& es) final;
  void interpret(l1t::HGCalMulticlusterBxCollection& multiclusters) const final;

private:
  std::vector<double> layer_containment_corrs_;
  std::vector<double> scale_corrections_coeff_;
  std::vector<double> dr_bylayer_;

  HGCalTriggerTools triggerTools_;
};

DEFINE_HGC_TPG_CLUSTER_INTERPRETER(HGCalTriggerClusterInterpretationEM, "HGCalTriggerClusterInterpretationEM");

HGCalTriggerClusterInterpretationEM::HGCalTriggerClusterInterpretationEM() {}

void HGCalTriggerClusterInterpretationEM::initialize(const edm::ParameterSet& conf) {
  layer_containment_corrs_ = conf.getParameter<std::vector<double>>("layer_containment_corrs");
  scale_corrections_coeff_ = conf.getParameter<std::vector<double>>("scale_correction_coeff");
  dr_bylayer_ = conf.getParameter<std::vector<double>>("dr_bylayer");

  const unsigned corrections_size = 2;
  if (scale_corrections_coeff_.size() != corrections_size) {
    throw cms::Exception("HGCTriggerParameterError")
        << "HGCalTriggerClusterInterpretationEM::scale_correction_coeff parameter has size: "
        << scale_corrections_coeff_.size() << " while expected is " << corrections_size;
  }
  if (layer_containment_corrs_.size() != dr_bylayer_.size()) {
    throw cms::Exception("HGCTriggerParameterError")
        << "HGCalTriggerClusterInterpretationEM::layer_containment_corrs and "
           "HGCalTriggerClusterInterpretationEM::dr_bylayer have different size!";
  }
}

void HGCalTriggerClusterInterpretationEM::eventSetup(const edm::EventSetup& es) { triggerTools_.eventSetup(es); }

void HGCalTriggerClusterInterpretationEM::interpret(l1t::HGCalMulticlusterBxCollection& multiclusters) const {
  for (unsigned int idx = 0; idx != multiclusters.size(); idx++) {
    l1t::HGCalMulticluster& cluster3d = multiclusters[idx];

    const GlobalPoint& cluster3d_position = cluster3d.centreProj();
    double energy = 0.;

    for (const auto& cluster2d : cluster3d.constituents()) {
      const unsigned layer = triggerTools_.triggerLayer(cluster2d.first);
      if (layer <= layer_containment_corrs_.size() - 1) {
        double dr = (cluster3d_position - cluster2d.second->centreProj()).mag();
        if (dr <= dr_bylayer_.at(layer)) {
          energy += layer_containment_corrs_.at(layer) * cluster2d.second->energy();
        }
      }
    }
    energy += scale_corrections_coeff_.at(1) * std::abs(cluster3d.eta()) + scale_corrections_coeff_.at(0);
    cluster3d.saveEnergyInterpretation(l1t::HGCalMulticluster::EnergyInterpretation::EM, max(energy, 0.));
  }
}

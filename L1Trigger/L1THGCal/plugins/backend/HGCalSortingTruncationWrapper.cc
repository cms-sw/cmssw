#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "L1Trigger/L1THGCal/interface/HGCalAlgoWrapperBase.h"
#include "L1Trigger/L1THGCal/interface/backend/HGCalSortingTruncationImpl_SA.h"

#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"
#include "L1Trigger/L1THGCal/interface/backend/HGCalMulticluster_SA.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerTools.h"

class HGCalSortingTruncationWrapper : public HGCalStage2FilteringWrapperBase {
public:
  HGCalSortingTruncationWrapper(const edm::ParameterSet& conf);
  ~HGCalSortingTruncationWrapper() override = default;

  void configure(
      const std::pair<const HGCalTriggerGeometryBase* const, const edm::ParameterSet&>& configuration) override;

  void process(const l1t::HGCalMulticlusterBxCollection&, l1t::HGCalMulticlusterBxCollection&) const override;

private:
  void convertCMSSWInputs(const l1t::HGCalMulticlusterBxCollection& multiclusters,
                          l1thgcfirmware::HGCalMulticlusterSACollection& multiclusters_SA) const;
  void convertAlgorithmOutputs(const l1thgcfirmware::HGCalMulticlusterSACollection& multiclusters_out,
                               const l1t::HGCalMulticlusterBxCollection& multiclusters_original,
                               l1t::HGCalMulticlusterBxCollection& multiclustersBXCollection) const;

  void setGeometry(const HGCalTriggerGeometryBase* const geom) { triggerTools_.setGeometry(geom); }

  HGCalTriggerTools triggerTools_;
  HGCalSortingTruncationImplSA theAlgo_;
  l1thgcfirmware::SortingTruncationAlgoConfig theConfiguration_;
};

HGCalSortingTruncationWrapper::HGCalSortingTruncationWrapper(const edm::ParameterSet& conf)
    : HGCalStage2FilteringWrapperBase(conf), theAlgo_(), theConfiguration_(conf.getParameter<unsigned>("maxTCs")) {}

void HGCalSortingTruncationWrapper::convertCMSSWInputs(
    const l1t::HGCalMulticlusterBxCollection& multiclusters,
    l1thgcfirmware::HGCalMulticlusterSACollection& multiclusters_SA) const {
  multiclusters_SA.reserve(multiclusters.size());

  for (unsigned int imulticluster = 0; imulticluster < multiclusters.size(); ++imulticluster) {
    bool firstConstituent = true;
    for (const auto& constituent : multiclusters[imulticluster].constituents()) {
      if (firstConstituent) {
        multiclusters_SA.emplace_back(l1thgcfirmware::HGCalMulticluster(
            l1thgcfirmware::HGCalCluster(constituent.second->centreProj().x(),
                                         constituent.second->centreProj().y(),
                                         constituent.second->centreProj().z(),
                                         triggerTools_.zside(constituent.second->detId()),
                                         triggerTools_.layerWithOffset(constituent.second->detId()),
                                         constituent.second->eta(),
                                         constituent.second->phi(),
                                         constituent.second->pt(),
                                         constituent.second->mipPt(),
                                         imulticluster),
            1.));

      } else {
        multiclusters_SA.at(imulticluster)
            .addConstituent(l1thgcfirmware::HGCalCluster(constituent.second->centreProj().x(),
                                                         constituent.second->centreProj().y(),
                                                         constituent.second->centreProj().z(),
                                                         triggerTools_.zside(constituent.second->detId()),
                                                         triggerTools_.layerWithOffset(constituent.second->detId()),
                                                         constituent.second->eta(),
                                                         constituent.second->phi(),
                                                         constituent.second->pt(),
                                                         constituent.second->mipPt(),
                                                         imulticluster),
                            1.);
      }
      firstConstituent = false;
    }
  }
}

void HGCalSortingTruncationWrapper::convertAlgorithmOutputs(
    const std::vector<l1thgcfirmware::HGCalMulticluster>& multiclusters_out,
    const l1t::HGCalMulticlusterBxCollection& multiclusters_original,
    l1t::HGCalMulticlusterBxCollection& multiclustersBXCollection) const {
  for (unsigned int imulticluster = 0; imulticluster < multiclusters_out.size(); ++imulticluster) {
    unsigned multicluster_id = multiclusters_out[imulticluster].constituents().at(0).index_cmssw();
    multiclustersBXCollection.push_back(0, multiclusters_original[multicluster_id]);
  }
}

void HGCalSortingTruncationWrapper::process(const l1t::HGCalMulticlusterBxCollection& inputMulticlusters,
                                            l1t::HGCalMulticlusterBxCollection& outputMulticlusters) const {
  l1thgcfirmware::HGCalMulticlusterSACollection multiclusters_SA;
  convertCMSSWInputs(inputMulticlusters, multiclusters_SA);

  l1thgcfirmware::HGCalMulticlusterSACollection multiclusters_finalized_SA;

  theAlgo_.sortAndTruncate_SA(multiclusters_SA, multiclusters_finalized_SA, theConfiguration_);

  convertAlgorithmOutputs(multiclusters_finalized_SA, inputMulticlusters, outputMulticlusters);
}

void HGCalSortingTruncationWrapper::configure(
    const std::pair<const HGCalTriggerGeometryBase* const, const edm::ParameterSet&>& configuration) {
  setGeometry(configuration.first);
};

DEFINE_EDM_PLUGIN(HGCalStage2FilteringWrapperBaseFactory,
                  HGCalSortingTruncationWrapper,
                  "HGCalSortingTruncationWrapper");

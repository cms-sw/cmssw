#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "L1Trigger/L1THGCal/interface/HGCalAlgoWrapperBase.h"
#include "L1Trigger/L1THGCal/interface/backend/HGCalTowerMapImpl_SA.h"

#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalTowerMap.h"
#include "DataFormats/L1THGCal/interface/HGCalTower.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "L1Trigger/L1THGCal/interface/backend/HGCalTowerMap2DImpl.h"
#include "L1Trigger/L1THGCal/interface/backend/HGCalTowerMap3DImpl.h"

#include "L1Trigger/L1THGCal/interface/backend/HGCalTower_SA.h"
#include "L1Trigger/L1THGCal/interface/backend/HGCalTowerMap_SA.h"
#include "L1Trigger/L1THGCal/interface/backend/HGCalTowerMapsConfig_SA.h"

class HGCalTowerMapsWrapper : public HGCalTowerMapsWrapperBase {
public:
  HGCalTowerMapsWrapper(const edm::ParameterSet& conf);
  ~HGCalTowerMapsWrapper() override = default;

  void configure(const std::pair<const HGCalTriggerGeometryBase* const, const edm::ParameterSet&>& parameters) override;

  void process(const std::vector<edm::Ptr<l1t::HGCalTowerMap>>& inputs,
               l1t::HGCalTowerBxCollection& outputs) const override;

private:
  void convertCMSSWInputs(const std::vector<edm::Ptr<l1t::HGCalTowerMap>>& inputTowerMaps,
                          std::vector<l1thgcfirmware::HGCalTowerMap>& towerMaps_SA) const;
  void convertAlgorithmOutputs(const std::vector<l1thgcfirmware::HGCalTower>& towerMaps_SA,
                               l1t::HGCalTowerBxCollection& outputTowerMaps) const;

  HGCalTowerMapImplSA theAlgo_;

  std::unique_ptr<l1thgcfirmware::TowerMapsAlgoConfig> theConfiguration_;
};

HGCalTowerMapsWrapper::HGCalTowerMapsWrapper(const edm::ParameterSet& conf) : HGCalTowerMapsWrapperBase(conf) {}

void HGCalTowerMapsWrapper::convertCMSSWInputs(const std::vector<edm::Ptr<l1t::HGCalTowerMap>>& inputTowerMaps,
                                               std::vector<l1thgcfirmware::HGCalTowerMap>& towerMaps_SA) const {
  for (const auto& map : inputTowerMaps) {
    std::vector<l1thgcfirmware::HGCalTowerCoord> tower_ids;
    for (const auto& tower : map->towers()) {
      tower_ids.emplace_back(tower.first, tower.second.eta(), tower.second.phi());
    }

    l1thgcfirmware::HGCalTowerMap towerMapSA(tower_ids);

    for (const auto& tower : map->towers()) {
      towerMapSA.addEt(tower.first, tower.second.etEm(), tower.second.etHad());
    }
    towerMaps_SA.emplace_back(towerMapSA);
  }
}

void HGCalTowerMapsWrapper::convertAlgorithmOutputs(const std::vector<l1thgcfirmware::HGCalTower>& towers_SA,
                                                    l1t::HGCalTowerBxCollection& outputTowerMaps) const {
  for (const auto& towerSA : towers_SA) {
    outputTowerMaps.push_back(
        0, l1t::HGCalTower(towerSA.etEm(), towerSA.etHad(), towerSA.eta(), towerSA.phi(), towerSA.id()));
  }
}

void HGCalTowerMapsWrapper::process(const std::vector<edm::Ptr<l1t::HGCalTowerMap>>& inputs,
                                    l1t::HGCalTowerBxCollection& outputs) const {
  std::vector<l1thgcfirmware::HGCalTowerMap> inputs_SA;
  convertCMSSWInputs(inputs, inputs_SA);

  std::vector<l1thgcfirmware::HGCalTower> outputs_SA;
  theAlgo_.runAlgorithm(inputs_SA, outputs_SA);

  convertAlgorithmOutputs(outputs_SA, outputs);
}

void HGCalTowerMapsWrapper::configure(
    const std::pair<const HGCalTriggerGeometryBase* const, const edm::ParameterSet&>& parameters) {}

DEFINE_EDM_PLUGIN(HGCalTowerMapsWrapperBaseFactory, HGCalTowerMapsWrapper, "HGCalTowerMapsWrapper");

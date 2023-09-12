// -*- C++ -*-
// Long Wang (UMD)
// plugin to run ML4DQM ONNX module and plot number of flagged bad channel counts vs LS
//

#include "DQM/HcalCommon/interface/DQTask.h"
#include "DQM/HcalCommon/interface/Utilities.h"
#include "DQM/HcalCommon/interface/HashFilter.h"
#include "DQM/HcalCommon/interface/Container1D.h"
#include "DQM/HcalCommon/interface/Container2D.h"
#include "DQM/HcalCommon/interface/ContainerProf1D.h"
#include "DQM/HcalCommon/interface/ContainerProf2D.h"
#include "DQM/HcalCommon/interface/ContainerSingle1D.h"
#include "DQM/HcalCommon/interface/ContainerSingle2D.h"
#include "DQM/HcalCommon/interface/ContainerSingleProf2D.h"
#include "DQM/HcalCommon/interface/ElectronicsMap.h"

#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"

#include "DQM/HcalTasks/plugins/OnlineDQMDigiAD_cmssw.h"

#include <cmath>
#include <iostream>
#include <algorithm>

using namespace cms::Ort;
using namespace hcaldqm;
using namespace hcaldqm::constants;
using namespace hcaldqm::filter;

class HcalMLTask : public hcaldqm::DQTask {
public:
  HcalMLTask(edm::ParameterSet const&);
  ~HcalMLTask() override = default;

  void dqmBeginRun(edm::Run const&, edm::EventSetup const&) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  std::shared_ptr<hcaldqm::Cache> globalBeginLuminosityBlock(edm::LuminosityBlock const&,
                                                             edm::EventSetup const&) const override;
  void globalEndLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void _process(edm::Event const&, edm::EventSetup const&) override;
  void _resetMonitors(hcaldqm::UpdateFreq) override;

  std::string onnx_model_path_HB, onnx_model_path_HE;
  double flagDecisionThr;
  edm::InputTag tagQIE11;
  edm::InputTag tagHO;
  edm::InputTag tagQIE10;
  edm::EDGetTokenT<QIE11DigiCollection> tokQIE11;
  edm::EDGetTokenT<HODigiCollection> tokHO;
  edm::EDGetTokenT<QIE10DigiCollection> tokQIE10;
  edm::ESGetToken<HcalDbService, HcalDbRecord> hcalDbServiceToken_;

  hcaldqm::ContainerXXX<double> Occupancy1LS;
  hcaldqm::Container1D MLFlagvsLS_Subdet;

  std::unique_ptr<OnlineDQMDigiAD> dqmadObj_HB = nullptr;
  std::unique_ptr<OnlineDQMDigiAD> dqmadObj_HE = nullptr;

  std::vector<std::vector<float>> digiHcal2DHist_depth_1{
      std::vector<std::vector<float>>(64, std::vector<float>(72, 0))};
  std::vector<std::vector<float>> digiHcal2DHist_depth_2{
      std::vector<std::vector<float>>(64, std::vector<float>(72, 0))};
  std::vector<std::vector<float>> digiHcal2DHist_depth_3{
      std::vector<std::vector<float>>(64, std::vector<float>(72, 0))};
  std::vector<std::vector<float>> digiHcal2DHist_depth_4{
      std::vector<std::vector<float>>(64, std::vector<float>(72, 0))};
  std::vector<std::vector<float>> digiHcal2DHist_depth_5{
      std::vector<std::vector<float>>(64, std::vector<float>(72, 0))};
  std::vector<std::vector<float>> digiHcal2DHist_depth_6{
      std::vector<std::vector<float>>(64, std::vector<float>(72, 0))};
  std::vector<std::vector<float>> digiHcal2DHist_depth_7{
      std::vector<std::vector<float>>(64, std::vector<float>(72, 0))};
};

HcalMLTask::HcalMLTask(edm::ParameterSet const& ps)
    : DQTask(ps), hcalDbServiceToken_(esConsumes<HcalDbService, HcalDbRecord, edm::Transition::BeginRun>()) {
  onnx_model_path_HB = ps.getUntrackedParameter<std::string>(
      "onnx_model_path_HB",
      "DQM/HcalTasks/data/HB_2022/"
      "CGAE_MultiDim_SPATIAL_vONNX_RCLv22_PIXEL_BT_BN_RIN_IPHI_MED_5218_v06_02_2023_21h01_stateful.onnx");
  onnx_model_path_HE = ps.getUntrackedParameter<std::string>(
      "onnx_model_path_HE",
      "DQM/HcalTasks/data/HE_2022/"
      "CGAE_MultiDim_SPATIAL_vONNX_RCLv22_PIXEL_BT_BN_RIN_IPHI_MED_7763_v06_02_2023_22h55_stateful.onnx");
  flagDecisionThr = ps.getUntrackedParameter<double>("flagDecisionThr", 20.);
  tagQIE11 = ps.getUntrackedParameter<edm::InputTag>("tagHBHE", edm::InputTag("hcalDigis"));
  tagHO = ps.getUntrackedParameter<edm::InputTag>("tagHO", edm::InputTag("hcalDigis"));
  tagQIE10 = ps.getUntrackedParameter<edm::InputTag>("tagHF", edm::InputTag("hcalDigis"));

  tokQIE11 = consumes<QIE11DigiCollection>(tagQIE11);
  tokHO = consumes<HODigiCollection>(tagHO);
  tokQIE10 = consumes<QIE10DigiCollection>(tagQIE10);

  auto dqmadObj_HB_ = std::make_unique<OnlineDQMDigiAD>("hb", onnx_model_path_HB, Backend::cpu);
  auto dqmadObj_HE_ = std::make_unique<OnlineDQMDigiAD>("he", onnx_model_path_HE, Backend::cpu);
  dqmadObj_HB = std::move(dqmadObj_HB_);
  dqmadObj_HE = std::move(dqmadObj_HE_);
}

void HcalMLTask::dqmBeginRun(edm::Run const& r, edm::EventSetup const& es) { DQTask::dqmBeginRun(r, es); }

void HcalMLTask::bookHistograms(DQMStore::IBooker& ib, edm::Run const& r, edm::EventSetup const& es) {
  DQTask::bookHistograms(ib, r, es);

  //	GET WHAT YOU NEED
  edm::ESHandle<HcalDbService> dbs = es.getHandle(hcalDbServiceToken_);
  _emap = dbs->getHcalMapping();

  //	Book monitoring elements
  Occupancy1LS.initialize(hcaldqm::hashfunctions::fDChannel);

  MLFlagvsLS_Subdet.initialize(_name,
                               "MLBadFlagedChannelsvsLS",
                               hcaldqm::hashfunctions::fSubdet,
                               new hcaldqm::quantity::LumiSection(_maxLS),
                               new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                               0);

  Occupancy1LS.book(_emap);
  MLFlagvsLS_Subdet.book(ib, _emap, _subsystem);
}

void HcalMLTask::_resetMonitors(hcaldqm::UpdateFreq uf) { DQTask::_resetMonitors(uf); }

void HcalMLTask::_process(edm::Event const& e, edm::EventSetup const&) {
  if (_ptype != fOnline)
    return;

  auto const chbhe = e.getHandle(tokQIE11);

  if (not(chbhe.isValid())) {
    edm::LogWarning("HcalMLTask") << "QIE11 Collection is unavailable, will not fill this event.";
    return;
  }

  auto lumiCache = luminosityBlockCache(e.getLuminosityBlock().index());
  _currentLS = lumiCache->currentLS;

  for (QIE11DigiCollection::const_iterator it = chbhe->begin(); it != chbhe->end(); ++it) {
    const QIE11DataFrame digi = static_cast<const QIE11DataFrame>(*it);

    HcalDetId const& did = digi.detid();
    if (did.subdet() != HcalEndcap && did.subdet() != HcalBarrel)
      continue;

    Occupancy1LS.get(did)++;
  }
}

std::shared_ptr<hcaldqm::Cache> HcalMLTask::globalBeginLuminosityBlock(edm::LuminosityBlock const& lb,
                                                                       edm::EventSetup const& es) const {
  return DQTask::globalBeginLuminosityBlock(lb, es);
}

void HcalMLTask::globalEndLuminosityBlock(edm::LuminosityBlock const& lb, edm::EventSetup const& es) {
  auto lumiCache = luminosityBlockCache(lb.index());
  _currentLS = lumiCache->currentLS;
  _xQuality.reset();
  _xQuality = lumiCache->xQuality;

  for (auto& HistElement : digiHcal2DHist_depth_1)
    std::fill(HistElement.begin(), HistElement.end(), 0);
  for (auto& HistElement : digiHcal2DHist_depth_2)
    std::fill(HistElement.begin(), HistElement.end(), 0);
  for (auto& HistElement : digiHcal2DHist_depth_3)
    std::fill(HistElement.begin(), HistElement.end(), 0);
  for (auto& HistElement : digiHcal2DHist_depth_4)
    std::fill(HistElement.begin(), HistElement.end(), 0);
  for (auto& HistElement : digiHcal2DHist_depth_5)
    std::fill(HistElement.begin(), HistElement.end(), 0);
  for (auto& HistElement : digiHcal2DHist_depth_6)
    std::fill(HistElement.begin(), HistElement.end(), 0);
  for (auto& HistElement : digiHcal2DHist_depth_7)
    std::fill(HistElement.begin(), HistElement.end(), 0);
  float LS_numEvents = (float)_evsPerLS;

  std::vector<HcalGenericDetId> dids = _emap->allPrecisionId();
  for (std::vector<HcalGenericDetId>::const_iterator it = dids.begin(); it != dids.end(); ++it) {
    if (!it->isHcalDetId())
      continue;
    if (_xQuality.exists(HcalDetId(*it))) {
      HcalChannelStatus cs(it->rawId(), _xQuality.get(HcalDetId(*it)));
      if (cs.isBitSet(HcalChannelStatus::HcalCellMask) || cs.isBitSet(HcalChannelStatus::HcalCellDead))
        continue;
    }

    HcalDetId did = HcalDetId(it->rawId());
    if (did.subdet() != HcalEndcap && did.subdet() != HcalBarrel)
      continue;

    if (did.depth() == 1)
      digiHcal2DHist_depth_1.at(did.ieta() < 0 ? did.ieta() + 32 : did.ieta() + 31).at(did.iphi() - 1) =
          Occupancy1LS.get(did);
    if (did.depth() == 2)
      digiHcal2DHist_depth_2.at(did.ieta() < 0 ? did.ieta() + 32 : did.ieta() + 31).at(did.iphi() - 1) =
          Occupancy1LS.get(did);
    if (did.depth() == 3)
      digiHcal2DHist_depth_3.at(did.ieta() < 0 ? did.ieta() + 32 : did.ieta() + 31).at(did.iphi() - 1) =
          Occupancy1LS.get(did);
    if (did.depth() == 4)
      digiHcal2DHist_depth_4.at(did.ieta() < 0 ? did.ieta() + 32 : did.ieta() + 31).at(did.iphi() - 1) =
          Occupancy1LS.get(did);
    if (did.depth() == 5)
      digiHcal2DHist_depth_5.at(did.ieta() < 0 ? did.ieta() + 32 : did.ieta() + 31).at(did.iphi() - 1) =
          Occupancy1LS.get(did);
    if (did.depth() == 6)
      digiHcal2DHist_depth_6.at(did.ieta() < 0 ? did.ieta() + 32 : did.ieta() + 31).at(did.iphi() - 1) =
          Occupancy1LS.get(did);
    if (did.depth() == 7)
      digiHcal2DHist_depth_7.at(did.ieta() < 0 ? did.ieta() + 32 : did.ieta() + 31).at(did.iphi() - 1) =
          Occupancy1LS.get(did);
  }

  std::vector<std::vector<float>> ad_HBmodel_output_vectors = dqmadObj_HB->Inference_CMSSW(digiHcal2DHist_depth_1,
                                                                                           digiHcal2DHist_depth_2,
                                                                                           digiHcal2DHist_depth_3,
                                                                                           digiHcal2DHist_depth_4,
                                                                                           digiHcal2DHist_depth_5,
                                                                                           digiHcal2DHist_depth_6,
                                                                                           digiHcal2DHist_depth_7,
                                                                                           LS_numEvents,
                                                                                           (float)flagDecisionThr);

  std::vector<std::vector<float>> ad_HEmodel_output_vectors = dqmadObj_HE->Inference_CMSSW(digiHcal2DHist_depth_1,
                                                                                           digiHcal2DHist_depth_2,
                                                                                           digiHcal2DHist_depth_3,
                                                                                           digiHcal2DHist_depth_4,
                                                                                           digiHcal2DHist_depth_5,
                                                                                           digiHcal2DHist_depth_6,
                                                                                           digiHcal2DHist_depth_7,
                                                                                           LS_numEvents,
                                                                                           (float)flagDecisionThr);

  std::vector<std::vector<std::vector<float>>> digiHcal3DHist_ANOMALY_FLAG_HB =
      dqmadObj_HB->ONNXOutputToDQMHistMap(ad_HBmodel_output_vectors, 4, 64, 7);
  std::vector<std::vector<std::vector<float>>> digiHcal3DHist_ANOMALY_FLAG_HE =
      dqmadObj_HE->ONNXOutputToDQMHistMap(ad_HEmodel_output_vectors, 7, 64, 7);

  int NHB_MLbadflags_ = 0, NHE_MLbadflags_ = 0;
  for (const auto& plane : digiHcal3DHist_ANOMALY_FLAG_HB)
    for (const auto& row : plane)
      NHB_MLbadflags_ += std::count(row.begin(), row.end(), 1);
  for (const auto& plane : digiHcal3DHist_ANOMALY_FLAG_HE)
    for (const auto& row : plane)
      NHE_MLbadflags_ += std::count(row.begin(), row.end(), 1);

  MLFlagvsLS_Subdet.fill(HcalDetId(HcalBarrel, 1, 1, 1), _currentLS, NHB_MLbadflags_);
  MLFlagvsLS_Subdet.fill(HcalDetId(HcalEndcap, 17, 1, 1), _currentLS, NHE_MLbadflags_);

  Occupancy1LS.reset();
  DQTask::globalEndLuminosityBlock(lb, es);
}

void HcalMLTask::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::string>("name", "HcalMLTask");
  desc.addUntracked<std::string>(
      "onnx_model_path_HB",
      "DQM/HcalTasks/data/HB_2022/"
      "CGAE_MultiDim_SPATIAL_vONNX_RCLv22_PIXEL_BT_BN_RIN_IPHI_MED_5218_v06_02_2023_21h01_stateful.onnx");
  desc.addUntracked<std::string>(
      "onnx_model_path_HE",
      "DQM/HcalTasks/data/HE_2022/"
      "CGAE_MultiDim_SPATIAL_vONNX_RCLv22_PIXEL_BT_BN_RIN_IPHI_MED_7763_v06_02_2023_22h55_stateful.onnx");
  desc.addUntracked<double>("flagDecisionThr", 20.);
  desc.addUntracked<int>("debug", 0);
  desc.addUntracked<int>("runkeyVal", 0);
  desc.addUntracked<std::string>("runkeyName", "pp_run");
  desc.addUntracked<int>("ptype", 1);
  desc.addUntracked<bool>("mtype", true);
  desc.addUntracked<std::string>("subsystem", "Hcal");
  desc.addUntracked<edm::InputTag>("tagHBHE", edm::InputTag("hcalDigis"));
  desc.addUntracked<edm::InputTag>("tagHO", edm::InputTag("hcalDigis"));
  desc.addUntracked<edm::InputTag>("tagHF", edm::InputTag("hcalDigis"));
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(HcalMLTask);

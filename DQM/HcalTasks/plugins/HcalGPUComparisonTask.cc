// -*- C++ -*-
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

#include <cmath>

using namespace hcaldqm;
using namespace hcaldqm::constants;
using namespace hcaldqm::filter;

class HcalGPUComparisonTask : public hcaldqm::DQTask {
public:
  HcalGPUComparisonTask(edm::ParameterSet const&);
  ~HcalGPUComparisonTask() override = default;

  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  std::shared_ptr<hcaldqm::Cache> globalBeginLuminosityBlock(edm::LuminosityBlock const&,
                                                             edm::EventSetup const&) const override;
  void globalEndLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void _process(edm::Event const&, edm::EventSetup const&) override;
  void _resetMonitors(hcaldqm::UpdateFreq) override;

  edm::InputTag tagHBHE_ref_;     //CPU version
  edm::InputTag tagHBHE_target_;  //GPU version
  edm::EDGetTokenT<HBHERecHitCollection> tokHBHE_ref_;
  edm::EDGetTokenT<HBHERecHitCollection> tokHBHE_target_;
  edm::ESGetToken<HcalDbService, HcalDbRecord> hcalDbServiceToken_;

  //    GPU reco test verification
  hcaldqm::Container2D energyGPUvsCPU_subdet_;
  hcaldqm::Container1D energyDiffGPUCPU_subdet_;
  hcaldqm::ContainerProf2D energyDiffGPUCPU_depth_;
};

HcalGPUComparisonTask::HcalGPUComparisonTask(edm::ParameterSet const& ps)
    : DQTask(ps),
      tagHBHE_ref_(ps.getUntrackedParameter<edm::InputTag>("tagHBHE_ref", edm::InputTag("hltHbhereco@cpu"))),
      tagHBHE_target_(ps.getUntrackedParameter<edm::InputTag>("tagHBHE_target", edm::InputTag("hltHbhereco@cuda"))),
      tokHBHE_ref_(consumes<HBHERecHitCollection>(tagHBHE_ref_)),
      tokHBHE_target_(consumes<HBHERecHitCollection>(tagHBHE_target_)),
      hcalDbServiceToken_(esConsumes<HcalDbService, HcalDbRecord, edm::Transition::BeginRun>()) {}

/* virtual */ void HcalGPUComparisonTask::bookHistograms(DQMStore::IBooker& ib,
                                                         edm::Run const& r,
                                                         edm::EventSetup const& es) {
  DQTask::bookHistograms(ib, r, es);

  //	GET WHAT YOU NEED
  edm::ESHandle<HcalDbService> dbs = es.getHandle(hcalDbServiceToken_);
  _emap = dbs->getHcalMapping();

  //	Book monitoring elements
  energyGPUvsCPU_subdet_.initialize(_name,
                                    "EnergyGPUvsCPU",
                                    hcaldqm::hashfunctions::fSubdet,
                                    new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fCPUenergy, true),
                                    new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fGPUenergy, true),
                                    new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                                    0);
  energyDiffGPUCPU_subdet_.initialize(_name,
                                      "EnergyDiffGPUCPU",
                                      hcaldqm::hashfunctions::fSubdet,
                                      new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fDiffRatio),
                                      new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),
                                      0);
  energyDiffGPUCPU_depth_.initialize(_name,
                                     "EnergyDiffGPUCPU",
                                     hcaldqm::hashfunctions::fdepth,
                                     new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fieta),
                                     new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fiphi),
                                     new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fDiffRatio),
                                     0);

  energyGPUvsCPU_subdet_.book(ib, _emap, _subsystem);
  energyDiffGPUCPU_subdet_.book(ib, _emap, _subsystem);
  energyDiffGPUCPU_depth_.book(ib, _emap, _subsystem);
}

/* virtual */ void HcalGPUComparisonTask::_resetMonitors(hcaldqm::UpdateFreq uf) { DQTask::_resetMonitors(uf); }

/* virtual */ void HcalGPUComparisonTask::_process(edm::Event const& e, edm::EventSetup const&) {
  edm::Handle<HBHERecHitCollection> chbhe_ref;
  edm::Handle<HBHERecHitCollection> chbhe_target;

  if (!(e.getByToken(tokHBHE_ref_, chbhe_ref)))
    _logger.dqmthrow("The CPU HBHERecHitCollection \"" + tagHBHE_ref_.encode() + "\" is not available");
  if (!(e.getByToken(tokHBHE_target_, chbhe_target)))
    _logger.dqmthrow("The GPU HBHERecHitCollection \"" + tagHBHE_target_.encode() + "\" is not available");

  auto lumiCache = luminosityBlockCache(e.getLuminosityBlock().index());
  _currentLS = lumiCache->currentLS;

  std::map<HcalDetId, double> mRecHitEnergy;

  for (HBHERecHitCollection::const_iterator it = chbhe_ref->begin(); it != chbhe_ref->end(); ++it) {
    double energy = it->energy();

    //	Explicit check on the DetIds present in the Collection
    HcalDetId did = it->id();

    if (mRecHitEnergy.find(did) == mRecHitEnergy.end())
      mRecHitEnergy.insert(std::make_pair(did, energy));
    else
      edm::LogError("HcalDQM|RechitTask") << "Duplicate Rechit from the same HcalDetId";
  }

  for (HBHERecHitCollection::const_iterator it = chbhe_target->begin(); it != chbhe_target->end(); ++it) {
    double energy = it->energy();
    HcalDetId did = it->id();

    if (mRecHitEnergy.find(did) != mRecHitEnergy.end()) {
      energyGPUvsCPU_subdet_.fill(did, mRecHitEnergy[did], energy);

      if (mRecHitEnergy[did] != 0.) {
        energyDiffGPUCPU_subdet_.fill(did, (energy - mRecHitEnergy[did]) / mRecHitEnergy[did]);
        if (energy > 0.1)
          energyDiffGPUCPU_depth_.fill(did, (energy - mRecHitEnergy[did]) / mRecHitEnergy[did]);
      } else if (mRecHitEnergy[did] == 0. && energy == 0.) {
        energyDiffGPUCPU_subdet_.fill(did, 0.);
        if (energy > 0.1)
          energyDiffGPUCPU_depth_.fill(did, 0.);
      } else {
        energyDiffGPUCPU_subdet_.fill(did, -1.);
        if (energy > 0.1)
          energyDiffGPUCPU_depth_.fill(did, -1.);
      }

      mRecHitEnergy.erase(did);
    } else
      edm::LogError("HcalDQM|RechitTask") << "GPU Rechit id not found in CPU Rechit id collection";
  }
  if (!mRecHitEnergy.empty())
    edm::LogError("HcalDQM|RechitTask") << "CPU Rechit id not found in GPU Rechit id collection";
}

std::shared_ptr<hcaldqm::Cache> HcalGPUComparisonTask::globalBeginLuminosityBlock(edm::LuminosityBlock const& lb,
                                                                                  edm::EventSetup const& es) const {
  return DQTask::globalBeginLuminosityBlock(lb, es);
}

/* virtual */ void HcalGPUComparisonTask::globalEndLuminosityBlock(edm::LuminosityBlock const& lb,
                                                                   edm::EventSetup const& es) {
  if (_ptype != fOnline)
    return;

  auto lumiCache = luminosityBlockCache(lb.index());
  _currentLS = lumiCache->currentLS;

  //	in the end always do the DQTask::endLumi
  DQTask::globalEndLuminosityBlock(lb, es);
}

void HcalGPUComparisonTask::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::string>("name", "HcalGPUComparisonTask");
  desc.addUntracked<int>("debug", 0);
  desc.addUntracked<int>("runkeyVal", 0);
  desc.addUntracked<std::string>("runkeyName", "pp_run");
  desc.addUntracked<int>("ptype", 1);
  desc.addUntracked<bool>("mtype", true);
  desc.addUntracked<std::string>("subsystem", "Hcal");
  desc.addUntracked<edm::InputTag>("tagHBHE_ref", edm::InputTag("hbhereco@cpu"));
  desc.addUntracked<edm::InputTag>("tagHBHE_target", edm::InputTag("hbhereco@cuda"));
  desc.addUntracked<edm::InputTag>("tagRaw", edm::InputTag("rawDataCollector"));
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(HcalGPUComparisonTask);

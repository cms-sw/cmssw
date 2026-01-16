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
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFraction.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include <cmath>
#ifdef PFLOW_DEBUG
#define LOGVERB(x) edm::LogVerbatim(x)
#else
#define LOGVERB(x) LogTrace(x)
#endif

using namespace hcaldqm;
using namespace hcaldqm::constants;
using namespace hcaldqm::filter;

class PFHcalGPUComparisonTask : public hcaldqm::DQTask {
public:
  PFHcalGPUComparisonTask(edm::ParameterSet const&);
  ~PFHcalGPUComparisonTask() override = default;

  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  std::shared_ptr<hcaldqm::Cache> globalBeginLuminosityBlock(edm::LuminosityBlock const&,
                                                             edm::EventSetup const&) const override;
  void globalEndLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void _process(edm::Event const&, edm::EventSetup const&) override;
  void _resetMonitors(hcaldqm::UpdateFreq) override;

  edm::EDGetTokenT<reco::PFClusterCollection> pfClusterTok_ref_;
  edm::EDGetTokenT<reco::PFClusterCollection> pfClusterTok_target_;

  MonitorElement* pfCluster_Multiplicity_HostvsDevice_;
  MonitorElement* pfCluster_Energy_HostvsDevice_;
  MonitorElement* pfCluster_RecHitMultiplicity_HostvsDevice_;
  MonitorElement* pfCluster_Layer_HostvsDevice_;
  MonitorElement* pfCluster_Depth_HostvsDevice_;
  MonitorElement* pfCluster_Eta_HostvsDevice_;
  MonitorElement* pfCluster_Phi_HostvsDevice_;
  MonitorElement* pfCluster_DuplicateMatches_HostvsDevice_;

  MonitorElement* pfCluster_Multiplicity_Diff_HostvsDevice_;
  MonitorElement* pfCluster_Energy_Diff_HostvsDevice_;
  MonitorElement* pfCluster_RecHitMultiplicity_Diff_HostvsDevice_;
  MonitorElement* pfCluster_Layer_Diff_HostvsDevice_;
  MonitorElement* pfCluster_Depth_Diff_HostvsDevice_;
  MonitorElement* pfCluster_Eta_Diff_HostvsDevice_;
  MonitorElement* pfCluster_Phi_Diff_HostvsDevice_;

  std::string subsystemDir_;
  std::string pfCaloGPUCompDir_;
};

PFHcalGPUComparisonTask::PFHcalGPUComparisonTask(edm::ParameterSet const& conf)
    : DQTask(conf),
      pfClusterTok_ref_{
          consumes<reco::PFClusterCollection>(conf.getUntrackedParameter<edm::InputTag>("pfClusterToken_ref"))},
      pfClusterTok_target_{
          consumes<reco::PFClusterCollection>(conf.getUntrackedParameter<edm::InputTag>("pfClusterToken_target"))},
      subsystemDir_{conf.getUntrackedParameter<std::string>("subsystem")},
      pfCaloGPUCompDir_{conf.getUntrackedParameter<std::string>("name")} {}

void PFHcalGPUComparisonTask::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& r, edm::EventSetup const& es) {
  _subsystem = subsystemDir_;
  ibooker.setCurrentFolder(pfCaloGPUCompDir_);
  DQTask::bookHistograms(ibooker, r, es);
  //	Book monitoring elements
  const char* histo;

  histo = "pfCluster_Multiplicity_HostvsDevice";
  const char* histoAxis = "pfCluster_Multiplicity_HostvsDevice;Multiplicity Device;Multiplicity Device";
  pfCluster_Multiplicity_HostvsDevice_ = ibooker.book2I(histo, histoAxis, 1000, 0, 1000, 1000, 0, 1000);

  histo = "pfCluster_Energy_HostvsDevice";
  histoAxis = "pfCluster_Energy_HostvsDevice;Energy Host [GeV];Energy Device [GeV]";
  pfCluster_Energy_HostvsDevice_ = ibooker.book2D(histo, histoAxis, 500, 0, 500, 500, 0, 500);

  histo = "pfCluster_RecHitMultiplicity_HostvsDevice";
  histoAxis = "pfCluster_RecHitMultiplicity_HostvsDevice;RecHit Multiplicity Host;RecHit Multiplicity Device";
  pfCluster_RecHitMultiplicity_HostvsDevice_ = ibooker.book2I(histo, histoAxis, 100, 0, 100, 100, 0, 100);

  histo = "pfCluster_Layer_HostvsDevice";
  histoAxis = "pfCluster_Layer_HostvsDevice;Cluster Layer Host;Cluster Layer Device";
  pfCluster_Layer_HostvsDevice_ = ibooker.book2I(histo, histoAxis, 4, 0, 3, 4, 0, 3);

  histo = "pfCluster_Depth_HostvsDevice";
  histoAxis = "pfCluster_Depth_HostvsDevice;Cluster Depth Host;Cluster Depth Device";
  pfCluster_Depth_HostvsDevice_ = ibooker.book2I(histo, histoAxis, 8, 0, 7, 8, 0, 7);

  histo = "pfCluster_Eta_HostvsDevice";
  histoAxis = "pfCluster_Eta_HostvsDevice;Cluster #eta Host;Cluster #eta Device";
  pfCluster_Eta_HostvsDevice_ = ibooker.book2D(histo, histoAxis, 100, -5.f, 5.f, 100, -5.f, 5.f);

  histo = "pfCluster_Phi_HostvsDevice";
  histoAxis = "pfCluster_Phi_HostvsDevice;Cluster #phi Host;Cluster #phi Device";
  pfCluster_Phi_HostvsDevice_ = ibooker.book2D(histo, histoAxis, 100, -M_PI, M_PI, 100, -M_PI, M_PI);

  histo = "pfCluster_DuplicateMatches_HostvsDevice";
  histoAxis = "pfCluster_Duplicates_HostvsDevice;Cluster Duplicates Host;Cluster Duplicates Device";
  pfCluster_DuplicateMatches_HostvsDevice_ = ibooker.book1I(histo, histoAxis, 100, 0., 1000);

  pfCluster_Multiplicity_Diff_HostvsDevice_ = ibooker.book1D(
      "MultiplicityDiff", "PFCluster Multiplicity Difference; (Reference - Target);#entries", 100, -2, 2);
  pfCluster_Energy_Diff_HostvsDevice_ =
      ibooker.book1D("EnergyDiff", "PFCluster Energy Difference; (Reference - Target);#entries", 100, -2, 2);
  pfCluster_RecHitMultiplicity_Diff_HostvsDevice_ = ibooker.book1D(
      "RHMultiplicityDiff", "PFCluster RecHit Multiplicity Difference; (Reference - Target);#entries", 100, -2, 2);
  pfCluster_Layer_Diff_HostvsDevice_ =
      ibooker.book1D("LayerDiff", "PFCluster Layer Difference; (Reference - Target);#entries", 100, -2, 2);
  pfCluster_Depth_Diff_HostvsDevice_ =
      ibooker.book1D("DepthDiff", "PFCluster Depth Difference; (Reference - Target);#entries", 100, -2, 2);
  pfCluster_Eta_Diff_HostvsDevice_ =
      ibooker.book1D("EtaDiff", "PFCluster #eta Difference; (Reference - Target);#entries", 100, -0.5, 0.5);
  pfCluster_Phi_Diff_HostvsDevice_ =
      ibooker.book1D("PhiDiff", "PFCluster #phi Difference; (Reference - Target);#entries", 100, -0.5, 0.5);
}

void PFHcalGPUComparisonTask::_resetMonitors(hcaldqm::UpdateFreq uf) { DQTask::_resetMonitors(uf); }

void PFHcalGPUComparisonTask::_process(edm::Event const& event, edm::EventSetup const&) {
  const auto& pfClusters_ref = event.getHandle(pfClusterTok_ref_);
  const auto& pfClusters_target = event.getHandle(pfClusterTok_target_);

  // Exit early if any handle is invalid
  if (!pfClusters_ref || !pfClusters_target) {
    edm::LogWarning out("PFHcalGPUComparisonTask");
    if (!pfClusters_ref)
      out << "reference PF cluster collection not found; ";
    if (!pfClusters_target)
      out << "target PF cluster collection not found; ";
    out << "the comparison will not run.";
    return;
  }

  auto lumiCache = luminosityBlockCache(event.getLuminosityBlock().index());
  _currentLS = lumiCache->currentLS;
  // Compare per-event PF cluster multiplicity

  if (pfClusters_ref->size() != pfClusters_target->size())
    LOGVERB("PFCaloGPUComparisonTask") << " PFCluster multiplicity " << pfClusters_ref->size() << " "
                                       << pfClusters_target->size();
  pfCluster_Multiplicity_HostvsDevice_->Fill((float)pfClusters_ref->size(), (float)pfClusters_target->size());
  pfCluster_Multiplicity_Diff_HostvsDevice_->Fill((float)pfClusters_ref->size() - (float)pfClusters_target->size());
  //
  // Find matching PF cluster pairs
  std::vector<int> matched_idx;
  matched_idx.reserve(pfClusters_ref->size());
  for (unsigned i = 0; i < pfClusters_ref->size(); ++i) {
    bool matched = false;
    for (unsigned j = 0; j < pfClusters_target->size(); ++j) {
      if (pfClusters_ref->at(i).seed() == pfClusters_target->at(j).seed()) {
        if (!matched) {
          matched = true;
          matched_idx.push_back((int)j);
        } else {
          edm::LogWarning("PFCaloGPUComparisonTask") << "Found duplicate match";
          pfCluster_DuplicateMatches_HostvsDevice_->Fill((int)j);
        }
      }
    }
    if (!matched)
      matched_idx.push_back(-1);  // if you don't find a match, put a dummy number
  }

  //
  // Plot matching PF cluster variables
  for (unsigned i = 0; i < pfClusters_ref->size(); ++i) {
    if (matched_idx[i] >= 0) {
      unsigned int j = matched_idx[i];
      int ref_energy_bin =
          pfCluster_Energy_HostvsDevice_->getTH2F()->GetXaxis()->FindBin(pfClusters_ref->at(i).energy());
      int target_energy_bin =
          pfCluster_Energy_HostvsDevice_->getTH2F()->GetXaxis()->FindBin(pfClusters_target->at(j).energy());
      if (ref_energy_bin != target_energy_bin)
        edm::LogPrint("PFCaloGPUComparisonTask")
            << "Off-diagonal energy bin entries: " << pfClusters_ref->at(i).energy() << " "
            << pfClusters_ref->at(i).eta() << " " << pfClusters_ref->at(i).phi() << " "
            << pfClusters_target->at(j).energy() << " " << pfClusters_target->at(j).eta() << " "
            << pfClusters_target->at(j).phi() << std::endl;
      pfCluster_Energy_HostvsDevice_->Fill(pfClusters_ref->at(i).energy(), pfClusters_target->at(j).energy());
      pfCluster_Layer_HostvsDevice_->Fill(pfClusters_ref->at(i).layer(), pfClusters_target->at(j).layer());
      pfCluster_Eta_HostvsDevice_->Fill(pfClusters_ref->at(i).eta(), pfClusters_target->at(j).eta());
      pfCluster_Phi_HostvsDevice_->Fill(pfClusters_ref->at(i).phi(), pfClusters_target->at(j).phi());
      pfCluster_Depth_HostvsDevice_->Fill(pfClusters_ref->at(i).depth(), pfClusters_target->at(j).depth());
      pfCluster_RecHitMultiplicity_HostvsDevice_->Fill((float)pfClusters_ref->at(i).recHitFractions().size(),
                                                       (float)pfClusters_target->at(j).recHitFractions().size());
      pfCluster_Energy_Diff_HostvsDevice_->Fill(pfClusters_ref->at(i).energy() - pfClusters_target->at(j).energy());
      pfCluster_RecHitMultiplicity_Diff_HostvsDevice_->Fill((float)pfClusters_ref->at(i).recHitFractions().size() -
                                                            (float)pfClusters_target->at(j).recHitFractions().size());
      pfCluster_Layer_Diff_HostvsDevice_->Fill(pfClusters_ref->at(i).layer() - pfClusters_target->at(j).layer());
      pfCluster_Depth_Diff_HostvsDevice_->Fill(pfClusters_ref->at(i).depth() - pfClusters_target->at(j).depth());
      ;
      pfCluster_Eta_Diff_HostvsDevice_->Fill(pfClusters_ref->at(i).eta() - pfClusters_target->at(j).eta());
      pfCluster_Phi_Diff_HostvsDevice_->Fill(
          reco::deltaPhi(pfClusters_ref->at(i).phi(), pfClusters_target->at(j).phi()));
    }
  }
}

std::shared_ptr<hcaldqm::Cache> PFHcalGPUComparisonTask::globalBeginLuminosityBlock(edm::LuminosityBlock const& lb,
                                                                                    edm::EventSetup const& es) const {
  return DQTask::globalBeginLuminosityBlock(lb, es);
}

void PFHcalGPUComparisonTask::globalEndLuminosityBlock(edm::LuminosityBlock const& lb, edm::EventSetup const& es) {
  if (_ptype != fOnline)
    return;

  auto lumiCache = luminosityBlockCache(lb.index());
  _currentLS = lumiCache->currentLS;

  //	in the end always do the DQTask::endLumi
  DQTask::globalEndLuminosityBlock(lb, es);
}

void PFHcalGPUComparisonTask::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::string>("subsystem", "ParticleFlow");
  desc.addUntracked<std::string>("name", "ParticleFlow/pfCaloGPUCompDir");
  desc.addUntracked<edm::InputTag>("pfClusterToken_ref", edm::InputTag("hltParticleFlowClusterHCALSerialSync"));
  desc.addUntracked<edm::InputTag>("pfClusterToken_target", edm::InputTag("hltParticleFlowClusterHCAL"));
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(PFHcalGPUComparisonTask);

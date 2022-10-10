#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1TParticleFlow/interface/PFCluster.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/corrector.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/ParametricResolution.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/HGC3DClusterEgID.h"
#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

namespace l1tpf {
  class PFClusterProducerFromHGC3DClusters : public edm::stream::EDProducer<> {
  public:
    explicit PFClusterProducerFromHGC3DClusters(const edm::ParameterSet &);
    ~PFClusterProducerFromHGC3DClusters() override {}

  private:
    enum class UseEmInterp { No, EmOnly, AllKeepHad, AllKeepTot };

    edm::EDGetTokenT<l1t::HGCalMulticlusterBxCollection> src_;
    UseEmInterp scenario_;
    bool emOnly_;
    double etCut_;
    StringCutObjectSelector<l1t::HGCalMulticluster> preEmId_;
    l1tpf::HGC3DClusterEgID emVsPionID_, emVsPUID_;
    bool hasEmId_;
    l1tpf::corrector corrector_;
    l1tpf::ParametricResolution resol_;

    void produce(edm::Event &, const edm::EventSetup &) override;

  };  // class
}  // namespace l1tpf

l1tpf::PFClusterProducerFromHGC3DClusters::PFClusterProducerFromHGC3DClusters(const edm::ParameterSet &iConfig)
    : src_(consumes<l1t::HGCalMulticlusterBxCollection>(iConfig.getParameter<edm::InputTag>("src"))),
      scenario_(UseEmInterp::No),
      emOnly_(iConfig.getParameter<bool>("emOnly")),
      etCut_(iConfig.getParameter<double>("etMin")),
      preEmId_(iConfig.getParameter<std::string>("preEmId")),
      emVsPionID_(iConfig.getParameter<edm::ParameterSet>("emVsPionID")),
      emVsPUID_(iConfig.getParameter<edm::ParameterSet>("emVsPUID")),
      hasEmId_((iConfig.existsAs<std::string>("preEmId") && !iConfig.getParameter<std::string>("preEmId").empty()) ||
               !emVsPionID_.method().empty()),
      corrector_(iConfig.getParameter<std::string>("corrector"),
                 emOnly_ || iConfig.getParameter<std::string>("corrector").empty()
                     ? -1
                     : iConfig.getParameter<double>("correctorEmfMax")),
      resol_(iConfig.getParameter<edm::ParameterSet>("resol")) {
  if (!emVsPionID_.method().empty()) {
    emVsPionID_.prepareTMVA();
  }
  if (!emVsPUID_.method().empty()) {
    emVsPUID_.prepareTMVA();
  }

  produces<l1t::PFClusterCollection>();
  produces<l1t::PFClusterCollection>("egamma");
  if (hasEmId_) {
    produces<l1t::PFClusterCollection>("em");
    produces<l1t::PFClusterCollection>("had");
  }

  std::string scenario = iConfig.getParameter<std::string>("useEMInterpretation");
  if (scenario == "emOnly") {
    scenario_ = UseEmInterp::EmOnly;
  } else if (scenario == "allKeepHad") {
    scenario_ = UseEmInterp::AllKeepHad;
    if (emOnly_) {
      throw cms::Exception("Configuration", "Unsupported emOnly = True when useEMInterpretation is " + scenario);
    }
  } else if (scenario == "allKeepTot") {
    scenario_ = UseEmInterp::AllKeepTot;
    if (emOnly_) {
      throw cms::Exception("Configuration", "Unsupported emOnly = True when useEMInterpretation is " + scenario);
    }
  } else if (scenario != "no") {
    throw cms::Exception("Configuration", "Unsupported useEMInterpretation scenario " + scenario);
  }
}

void l1tpf::PFClusterProducerFromHGC3DClusters::produce(edm::Event &iEvent, const edm::EventSetup &) {
  auto out = std::make_unique<l1t::PFClusterCollection>();
  auto outEgamma = std::make_unique<l1t::PFClusterCollection>();
  std::unique_ptr<l1t::PFClusterCollection> outEm, outHad;
  if (hasEmId_) {
    outEm = std::make_unique<l1t::PFClusterCollection>();
    outHad = std::make_unique<l1t::PFClusterCollection>();
  }
  edm::Handle<l1t::HGCalMulticlusterBxCollection> multiclusters;
  iEvent.getByToken(src_, multiclusters);

  for (auto it = multiclusters->begin(0), ed = multiclusters->end(0); it != ed; ++it) {
    float pt = it->pt(), hoe = it->hOverE();
    bool isEM = hasEmId_ ? preEmId_(*it) : emOnly_;
    if (emOnly_) {
      if (hoe == -1)
        continue;
      pt /= (1 + hoe);
      hoe = 0;
    }
    if (pt <= etCut_)
      continue;

    // this block below is to support the older EG emulators, and is not used in newer ones
    if (it->hwQual()) {  // this is the EG ID shipped with the HGC TPs
      // we use the EM interpretation of the cluster energy
      l1t::PFCluster egcluster(
          it->iPt(l1t::HGCalMulticluster::EnergyInterpretation::EM), it->eta(), it->phi(), hoe, false);
      egcluster.setHwQual(it->hwQual());
      egcluster.addConstituent(edm::Ptr<l1t::L1Candidate>(multiclusters, multiclusters->key(it)));
      outEgamma->push_back(egcluster);
    }

    l1t::PFCluster cluster(pt, it->eta(), it->phi(), hoe);
    if (scenario_ == UseEmInterp::EmOnly) {  // for emID objs, use EM interp as pT and set H = 0
      if (isEM) {
        float pt_new = it->iPt(l1t::HGCalMulticluster::EnergyInterpretation::EM);
        float hoe_new = 0.;
        cluster = l1t::PFCluster(pt_new, it->eta(), it->phi(), hoe_new, /*isEM=*/isEM);
      }
    } else if (scenario_ == UseEmInterp::AllKeepHad) {  // for all objs, replace EM part with EM interp, preserve H
      float had_old = pt - cluster.emEt();
      //float em_old = cluster.emEt();
      float em_new = it->iPt(l1t::HGCalMulticluster::EnergyInterpretation::EM);
      float pt_new = had_old + em_new;
      float hoe_new = em_new > 0 ? (had_old / em_new) : -1;
      cluster = l1t::PFCluster(pt_new, it->eta(), it->phi(), hoe_new, /*isEM=*/isEM);
      //printf("Scenario %d: pt %7.2f eta %+5.3f em %7.2f, EMI %7.2f, h/e % 8.3f --> pt %7.2f, em %7.2f, h/e % 8.3f\n",
      //        2, pt, it->eta(), em_old, em_new, hoe, cluster.pt(), cluster.emEt(), cluster.hOverE());
    } else if (scenario_ == UseEmInterp::AllKeepTot) {  // for all objs, replace EM part with EM interp, preserve pT
      //float em_old = cluster.emEt();
      float em_new = it->iPt(l1t::HGCalMulticluster::EnergyInterpretation::EM);
      float hoe_new = em_new > 0 ? (it->pt() / em_new - 1) : -1;
      cluster = l1t::PFCluster(it->pt(), it->eta(), it->phi(), hoe_new, /*isEM=*/isEM);
      //printf("Scenario %d: pt %7.2f eta %+5.3f em %7.2f, EMI %7.2f, h/e % 8.3f --> pt %7.2f, em %7.2f, h/e % 8.3f\n",
      //        3, pt, it->eta(), em_old, em_new, hoe, cluster.pt(), cluster.emEt(), cluster.hOverE());
    }

    if (!emVsPUID_.method().empty()) {
      if (!emVsPUID_.passID(*it, cluster)) {
        continue;
      }
    }
    if (!emOnly_ && !emVsPionID_.method().empty()) {
      isEM = emVsPionID_.passID(*it, cluster);
    }
    cluster.setHwQual((isEM ? 1 : 0) + (it->hwQual() << 1));

    if (corrector_.valid())
      corrector_.correctPt(cluster);
    cluster.setPtError(resol_(cluster.pt(), std::abs(cluster.eta())));

    // We se the cluster shape variables used downstream
    cluster.setAbsZBarycenter(fabs(it->zBarycenter()));
    cluster.setSigmaRR(it->sigmaRRTot());

    out->push_back(cluster);
    out->back().addConstituent(edm::Ptr<l1t::L1Candidate>(multiclusters, multiclusters->key(it)));
    if (hasEmId_) {
      (isEM ? outEm : outHad)->push_back(out->back());
    }
  }

  iEvent.put(std::move(out));
  iEvent.put(std::move(outEgamma), "egamma");
  if (hasEmId_) {
    iEvent.put(std::move(outEm), "em");
    iEvent.put(std::move(outHad), "had");
  }
}
using l1tpf::PFClusterProducerFromHGC3DClusters;
DEFINE_FWK_MODULE(PFClusterProducerFromHGC3DClusters);

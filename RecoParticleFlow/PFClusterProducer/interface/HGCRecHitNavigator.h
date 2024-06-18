#ifndef RecoParticleFlow_PFClusterProducer_HGCRecHitNavigator_h
#define RecoParticleFlow_PFClusterProducer_HGCRecHitNavigator_h

#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitNavigatorBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

template <ForwardSubdetector D1,
          typename hgcee,
          ForwardSubdetector D2,
          typename hgchef,
          ForwardSubdetector D3,
          typename hgcheb>
class HGCRecHitNavigator : public PFRecHitNavigatorBase {
public:
  HGCRecHitNavigator() = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;

    desc.add<std::string>("name", "PFRecHitHGCNavigator");

    edm::ParameterSetDescription descee;
    descee.add<std::string>("name", "PFRecHitHGCEENavigator");
    descee.add<std::string>("topologySource", "HGCalEESensitive");
    desc.add<edm::ParameterSetDescription>("hgcee", descee);

    edm::ParameterSetDescription deschef;
    deschef.add<std::string>("name", "PFRecHitHGCHENavigator");
    deschef.add<std::string>("topologySource", "HGCalHESiliconSensitive");
    desc.add<edm::ParameterSetDescription>("hgchef", deschef);

    edm::ParameterSetDescription descheb;
    descheb.add<std::string>("name", "PFRecHitHGCHENavigator");
    descheb.add<std::string>("topologySource", "HGCalHEScintillatorSensitive");
    desc.add<edm::ParameterSetDescription>("hgcheb", descheb);

    descriptions.add("navigator", desc);
  }

  HGCRecHitNavigator(const edm::ParameterSet& iConfig, edm::ConsumesCollector& cc) {
    const auto& pset_hgcee = iConfig.getParameter<edm::ParameterSet>("hgcee");
    if (!pset_hgcee.empty() && !pset_hgcee.getParameter<std::string>("name").empty()) {
      eeNav_ = new hgcee(pset_hgcee, cc);
    } else {
      eeNav_ = nullptr;
    }
    const auto& pset_hgchef = iConfig.getParameter<edm::ParameterSet>("hgchef");
    if (!pset_hgchef.empty() && !pset_hgchef.getParameter<std::string>("name").empty()) {
      hefNav_ = new hgchef(pset_hgchef, cc);
    } else {
      hefNav_ = nullptr;
    }
    const auto& pset_hgcheb = iConfig.getParameter<edm::ParameterSet>("hgcheb");
    if (!pset_hgcheb.empty() && !pset_hgcheb.getParameter<std::string>("name").empty()) {
      hebNav_ = new hgcheb(pset_hgcheb, cc);
    } else {
      hebNav_ = nullptr;
    }
  }

  void init(const edm::EventSetup& iSetup) override {
    if (nullptr != eeNav_)
      eeNav_->init(iSetup);
    if (nullptr != hefNav_)
      hefNav_->init(iSetup);
    if (nullptr != hebNav_)
      hebNav_->init(iSetup);
  }

  void associateNeighbours(reco::PFRecHit& hit,
                           std::unique_ptr<reco::PFRecHitCollection>& hits,
                           edm::RefProd<reco::PFRecHitCollection>& refProd) override {
    switch (DetId(hit.detId()).subdetId()) {
      case D1:
        if (nullptr != eeNav_)
          eeNav_->associateNeighbours(hit, hits, refProd);
        break;
      case D2:
        if (nullptr != hefNav_)
          hefNav_->associateNeighbours(hit, hits, refProd);
        break;
      case D3:
        if (nullptr != hebNav_)
          hebNav_->associateNeighbours(hit, hits, refProd);
        break;
      default:
        break;
    }
  }

protected:
  hgcee* eeNav_;
  hgchef* hefNav_;
  hgcheb* hebNav_;
};

#endif

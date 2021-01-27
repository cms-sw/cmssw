#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/Common/interface/View.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"
#include "SimDataFormats/CaloAnalysis/interface/SimClusterFwd.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticleFwd.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/Common/interface/Association.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include <vector>
#include <iostream>

template <typename T, typename M>
class ObjectIndexFromAssociationTableProducer : public edm::global::EDProducer<> {
public:
  ObjectIndexFromAssociationTableProducer(edm::ParameterSet const& params)
      : objName_(params.getParameter<std::string>("objName")),
        branchName_(params.getParameter<std::string>("branchName")),
        doc_(params.getParameter<std::string>("docString")),
        src_(consumes<T>(params.getParameter<edm::InputTag>("src"))),
        objMap_(consumes<edm::Association<M>>(params.getParameter<edm::InputTag>("objMap"))),
        cut_(params.getParameter<std::string>("cut"), true) {
    produces<nanoaod::FlatTable>();
  }

  ~ObjectIndexFromAssociationTableProducer() override {}

  void produce(edm::StreamID id, edm::Event& iEvent, const edm::EventSetup& iSetup) const override {
    edm::Handle<T> objs;
    iEvent.getByToken(src_, objs);

    edm::Handle<edm::Association<M>> assoc;
    iEvent.getByToken(objMap_, assoc);

    std::vector<int> keys;
    for (unsigned int i = 0; i < objs->size(); ++i) {
      edm::Ref<T> tk(objs, i);
      if (cut_(*tk)) {
        edm::Ref<M> match = (*assoc)[tk];
        int key = match.isNonnull() ? match.key() : -1;
        keys.emplace_back(key);
      }
    }

    auto tab = std::make_unique<nanoaod::FlatTable>(keys.size(), objName_, false, true);
    tab->addColumn<int>(branchName_ + "Idx", keys, doc_);

    iEvent.put(std::move(tab));
  }

protected:
  const std::string objName_, branchName_, doc_;
  const edm::EDGetTokenT<T> src_;
  const edm::EDGetTokenT<edm::Association<M>> objMap_;
  const StringCutObjectSelector<typename T::value_type> cut_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
typedef ObjectIndexFromAssociationTableProducer<edm::SimTrackContainer, SimClusterCollection> SimTrackToSimClusterIndexTableProducer;
typedef ObjectIndexFromAssociationTableProducer<edm::PCaloHitContainer, SimClusterCollection> CaloHitToSimClusterIndexTableProducer;
typedef ObjectIndexFromAssociationTableProducer<edm::View<CaloRecHit>, SimClusterCollection> CaloRecHitToSimClusterIndexTableProducer;
typedef ObjectIndexFromAssociationTableProducer<edm::View<CaloRecHit>, reco::PFCandidateCollection> CaloRecHitToPFCandIndexTableProducer;
typedef ObjectIndexFromAssociationTableProducer<SimClusterCollection, CaloParticleCollection> SimClusterToCaloParticleIndexTableProducer;
typedef ObjectIndexFromAssociationTableProducer<SimClusterCollection, SimClusterCollection> SimClusterToSimClusterIndexTableProducer;
DEFINE_FWK_MODULE(SimTrackToSimClusterIndexTableProducer);
DEFINE_FWK_MODULE(CaloHitToSimClusterIndexTableProducer);
DEFINE_FWK_MODULE(SimClusterToCaloParticleIndexTableProducer);
DEFINE_FWK_MODULE(CaloRecHitToSimClusterIndexTableProducer);
DEFINE_FWK_MODULE(SimClusterToSimClusterIndexTableProducer);
DEFINE_FWK_MODULE(CaloRecHitToPFCandIndexTableProducer);

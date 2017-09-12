#ifndef RecoPixelVertexing_PixelTriplets_HitTripletEDProducerT_H
#define RecoPixelVertexing_PixelTriplets_HitTripletEDProducerT_H

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Utilities/interface/RunningAverage.h"

#include "RecoTracker/TkHitPairs/interface/IntermediateHitDoublets.h"
#include "RecoTracker/TkHitPairs/interface/RegionsSeedingHitSets.h"
#include "RecoPixelVertexing/PixelTriplets/interface/OrderedHitTriplets.h"
#include "RecoPixelVertexing/PixelTriplets/interface/IntermediateHitTriplets.h"
#include "RecoPixelVertexing/PixelTriplets/interface/LayerTriplets.h"

#include <numeric>

namespace hitTripletEDProducerT { class ImplBase; }

template <typename T_HitTripletGenerator>
class HitTripletEDProducerT: public edm::stream::EDProducer<> {
public:
  HitTripletEDProducerT(const edm::ParameterSet& iConfig);
  ~HitTripletEDProducerT() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

private:
  edm::EDGetTokenT<IntermediateHitDoublets> doubletToken_;

  std::unique_ptr<hitTripletEDProducerT::ImplBase> impl_;
};

namespace hitTripletEDProducerT {
  class ImplBase {
  public:
    ImplBase() = default;
    virtual ~ImplBase() = default;

    virtual void produces(edm::ProducerBase& producer) const = 0;
    virtual void produce(const IntermediateHitDoublets& regionDoublets,
                         edm::Event& iEvent, const edm::EventSetup& iSetup) = 0;

  protected:
    edm::RunningAverage localRA_;
  };

  /////
  template <typename T_HitTripletGenerator>
  class ImplGeneratorBase: public ImplBase {
  public:
    ImplGeneratorBase(const edm::ParameterSet& iConfig, edm::ConsumesCollector& iC):
      generator_(iConfig, iC)
    {}
    ~ImplGeneratorBase() override = default;

  protected:
    T_HitTripletGenerator generator_;
  };

  /////
  template <typename T_HitTripletGenerator,
            typename T_SeedingHitSets, typename T_IntermediateHitTriplets>
  class Impl: public ImplGeneratorBase<T_HitTripletGenerator> {
  public:
    Impl(const edm::ParameterSet& iConfig, edm::ConsumesCollector& iC):
      ImplGeneratorBase<T_HitTripletGenerator>(iConfig, iC) {}
    ~Impl() override = default;

    void produces(edm::ProducerBase& producer) const override {
      T_SeedingHitSets::produces(producer);
      T_IntermediateHitTriplets::produces(producer);
    };

    void produce(const IntermediateHitDoublets& regionDoublets,
                 edm::Event& iEvent, const edm::EventSetup& iSetup) override {
      const SeedingLayerSetsHits& seedingLayerHits = regionDoublets.seedingLayerHits();

      auto seedingHitSetsProducer = T_SeedingHitSets();
      auto intermediateHitTripletsProducer = T_IntermediateHitTriplets(&seedingLayerHits);

      if(regionDoublets.empty()) {
        seedingHitSetsProducer.putEmpty(iEvent);
        intermediateHitTripletsProducer.putEmpty(iEvent);
        return;
      }

      seedingHitSetsProducer.reserve(regionDoublets.regionSize(), this->localRA_.upper());
      intermediateHitTripletsProducer.reserve(regionDoublets.regionSize(), this->localRA_.upper());

      // match-making of pair and triplet layers
      std::vector<LayerTriplets::LayerSetAndLayers> trilayers = LayerTriplets::layers(seedingLayerHits);

      OrderedHitTriplets triplets;
      triplets.reserve(this->localRA_.upper());
      size_t triplets_total = 0;

      LogDebug("HitTripletEDProducer") << "Creating triplets for " << regionDoublets.regionSize() << " regions, and " << trilayers.size() << " pair+3rd layers from " << regionDoublets.layerPairsSize() << " layer pairs";

      for(const auto& regionLayerPairs: regionDoublets) {
        const TrackingRegion& region = regionLayerPairs.region();

        auto hitCachePtr_filler_shs = seedingHitSetsProducer.beginRegion(&region, nullptr);
        auto hitCachePtr_filler_iht = intermediateHitTripletsProducer.beginRegion(&region, std::get<0>(hitCachePtr_filler_shs));
        auto hitCachePtr = std::get<0>(hitCachePtr_filler_iht);

        LayerHitMapCache& hitCache = *hitCachePtr;
        hitCache.extend(regionLayerPairs.layerHitMapCache());

        LogTrace("HitTripletEDProducer") << " starting region";

        for(const auto& layerPair: regionLayerPairs) {
          LogTrace("HitTripletEDProducer") << "  starting layer pair " << layerPair.innerLayerIndex() << "," << layerPair.outerLayerIndex();

          auto found = std::find_if(trilayers.begin(), trilayers.end(), [&](const LayerTriplets::LayerSetAndLayers& a) {
              return a.first[0].index() == layerPair.innerLayerIndex() && a.first[1].index() == layerPair.outerLayerIndex();
            });
          if(found == trilayers.end()) {
            auto exp = cms::Exception("LogicError") << "Did not find the layer pair from vector<pair+third layers>. This is a sign of some internal inconsistency\n";
            exp << "I was looking for layer pair " << layerPair.innerLayerIndex() << "," << layerPair.outerLayerIndex() << ". Triplets have the following pairs:\n";
            for(const auto& a: trilayers) {
              exp << " " << a.first[0].index() << "," << a.first[1].index() << ": 3rd layers";
              for(const auto& b: a.second) {
              exp << " " << b.index();
              }
              exp << "\n";
            }
            throw exp;
          }
          const auto& thirdLayers = found->second;

          this->generator_.hitTriplets(region, triplets, iEvent, iSetup, layerPair.doublets(), thirdLayers,
                                       intermediateHitTripletsProducer.tripletLastLayerIndexVector(), hitCache);

#ifdef EDM_ML_DEBUG
          LogTrace("HitTripletEDProducer") << "  created " << triplets.size() << " triplets for layer pair " << layerPair.innerLayerIndex() << "," << layerPair.outerLayerIndex() << " and 3rd layers";
          for(const auto& l: thirdLayers) {
            LogTrace("HitTripletEDProducer") << "   " << l.index();
          }
#endif

          triplets_total += triplets.size();
          seedingHitSetsProducer.fill(std::get<1>(hitCachePtr_filler_shs), triplets);
          intermediateHitTripletsProducer.fill(std::get<1>(hitCachePtr_filler_iht), layerPair.layerPair(), thirdLayers, triplets);

          triplets.clear();
        }
      }
      this->localRA_.update(triplets_total);

      seedingHitSetsProducer.put(iEvent);
      intermediateHitTripletsProducer.put(iEvent);
    }
  };


  /////
  class DoNothing {
  public:
    DoNothing() {}
    explicit DoNothing(const SeedingLayerSetsHits *layers) {}

    static void produces(edm::ProducerBase&) {}

    void reserve(size_t, size_t) {}

    auto beginRegion(const TrackingRegion *, LayerHitMapCache *ptr) {
      return std::make_tuple(ptr, 0);
    }

    std::vector<int> *tripletLastLayerIndexVector() {
      return nullptr;
    }

    void fill(int, const OrderedHitTriplets&) {}
    void fill(int, const IntermediateHitTriplets::LayerPair&,
              const std::vector<SeedingLayerSetsHits::SeedingLayer>&,
              const OrderedHitTriplets&) {}


    void put(edm::Event& iEvent) {}
    void putEmpty(edm::Event& iEvent) {}
  };

  /////
  class ImplSeedingHitSets {
  public:
    ImplSeedingHitSets():
      seedingHitSets_(std::make_unique<RegionsSeedingHitSets>())
    {}

    static void produces(edm::ProducerBase& producer) {
      producer.produces<RegionsSeedingHitSets>();
    }

    void reserve(size_t regionsSize, size_t localRAupper) {
      seedingHitSets_->reserve(regionsSize, localRAupper);
    }

    auto beginRegion(const TrackingRegion *region, LayerHitMapCache *ptr) {
      hitCacheTmp_.clear();
      return std::make_tuple(&hitCacheTmp_, seedingHitSets_->beginRegion(region));
    }

    void fill(RegionsSeedingHitSets::RegionFiller& filler, const OrderedHitTriplets& triplets) {
      for(const auto& trpl: triplets) {
        filler.emplace_back(trpl.inner(), trpl.middle(), trpl.outer());
      }
    }

    void put(edm::Event& iEvent) {
      seedingHitSets_->shrink_to_fit();
      putEmpty(iEvent);
    }
    void putEmpty(edm::Event& iEvent) {
      iEvent.put(std::move(seedingHitSets_));
    }

  private:
    std::unique_ptr<RegionsSeedingHitSets> seedingHitSets_;
    LayerHitMapCache hitCacheTmp_; // used if !produceIntermediateHitDoublets
  };

  /////
  class ImplIntermediateHitTriplets {
  public:
    explicit ImplIntermediateHitTriplets(const SeedingLayerSetsHits *layers):
      intermediateHitTriplets_(std::make_unique<IntermediateHitTriplets>(layers)),
      layers_(layers)
    {}

    static void produces(edm::ProducerBase& producer) {
      producer.produces<IntermediateHitTriplets>();
    }

    void reserve(size_t regionsSize, size_t localRAupper) {
      intermediateHitTriplets_->reserve(regionsSize, layers_->size(), localRAupper);
      tripletLastLayerIndex_.reserve(localRAupper);
    }

    auto beginRegion(const TrackingRegion *region, LayerHitMapCache *) {
      auto filler = intermediateHitTriplets_->beginRegion(region);
      return std::make_tuple(&(filler.layerHitMapCache()), std::move(filler));
    }

    std::vector<int> *tripletLastLayerIndexVector() {
      return &tripletLastLayerIndex_;
    }

    void fill(IntermediateHitTriplets::RegionFiller& filler,
              const IntermediateHitTriplets::LayerPair& layerPair,
              const std::vector<SeedingLayerSetsHits::SeedingLayer>& thirdLayers,
              const OrderedHitTriplets& triplets) {
      if(tripletLastLayerIndex_.size() != triplets.size()) {
        throw cms::Exception("LogicError") << "tripletLastLayerIndex_.size() " << tripletLastLayerIndex_.size()
                                           << " triplets.size() " << triplets.size();
      }
      tripletPermutation_.resize(tripletLastLayerIndex_.size());
      std::iota(tripletPermutation_.begin(), tripletPermutation_.end(), 0); // assign 0,1,2,...,N
      std::stable_sort(tripletPermutation_.begin(), tripletPermutation_.end(), [&](size_t i, size_t j) {
          return tripletLastLayerIndex_[i] < tripletLastLayerIndex_[j];
        });

      // empty triplets need to propagate here
      filler.addTriplets(layerPair, thirdLayers, triplets, tripletLastLayerIndex_, tripletPermutation_);
      tripletLastLayerIndex_.clear();
    }

    void put(edm::Event& iEvent) {
      intermediateHitTriplets_->shrink_to_fit();
      putEmpty(iEvent);
    }
    void putEmpty(edm::Event& iEvent) {
      iEvent.put(std::move(intermediateHitTriplets_));
    }

  private:
    std::unique_ptr<IntermediateHitTriplets> intermediateHitTriplets_;
    const SeedingLayerSetsHits *layers_;
    std::vector<size_t> tripletPermutation_;
    std::vector<int> tripletLastLayerIndex_;
  };
}



template <typename T_HitTripletGenerator>
HitTripletEDProducerT<T_HitTripletGenerator>::HitTripletEDProducerT(const edm::ParameterSet& iConfig):
  doubletToken_(consumes<IntermediateHitDoublets>(iConfig.getParameter<edm::InputTag>("doublets")))
{
  const bool produceSeedingHitSets = iConfig.getParameter<bool>("produceSeedingHitSets");
  const bool produceIntermediateHitTriplets = iConfig.getParameter<bool>("produceIntermediateHitTriplets");

  auto iC = consumesCollector();

  using namespace hitTripletEDProducerT;

  if(produceSeedingHitSets && produceIntermediateHitTriplets)
    impl_ = std::make_unique<Impl<T_HitTripletGenerator, ImplSeedingHitSets, ImplIntermediateHitTriplets>>(iConfig, iC);
  else if(produceSeedingHitSets)
    impl_ = std::make_unique<Impl<T_HitTripletGenerator, ImplSeedingHitSets, DoNothing>>(iConfig, iC);
  else if(produceIntermediateHitTriplets)
    impl_ = std::make_unique<Impl<T_HitTripletGenerator, DoNothing, ImplIntermediateHitTriplets>>(iConfig, iC);
  else
    throw cms::Exception("Configuration") << "HitTripletEDProducerT requires either produceIntermediateHitTriplets or produceSeedingHitSets to be True. If neither are needed, just remove this module from your sequence/path as it doesn't do anything useful";

  impl_->produces(*this);
}

template <typename T_HitTripletGenerator>
void HitTripletEDProducerT<T_HitTripletGenerator>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("doublets", edm::InputTag("hitPairEDProducer"));
  desc.add<bool>("produceSeedingHitSets", false);
  desc.add<bool>("produceIntermediateHitTriplets", false);

  T_HitTripletGenerator::fillDescriptions(desc);

  auto label = T_HitTripletGenerator::fillDescriptionsLabel() + std::string("EDProducerDefault");
  descriptions.add(label, desc);
}

template <typename T_HitTripletGenerator>
void HitTripletEDProducerT<T_HitTripletGenerator>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<IntermediateHitDoublets> hdoublets;
  iEvent.getByToken(doubletToken_, hdoublets);
  const auto& regionDoublets = *hdoublets;

  const SeedingLayerSetsHits& seedingLayerHits = regionDoublets.seedingLayerHits();
  if(seedingLayerHits.numberOfLayersInSet() < 3) {
    throw cms::Exception("LogicError") << "HitTripletEDProducerT expects SeedingLayerSetsHits::numberOfLayersInSet() to be >= 3, got " << seedingLayerHits.numberOfLayersInSet() << ". This is likely caused by a configuration error of this module, HitPairEDProducer, or SeedingLayersEDProducer.";
  }

  impl_->produce(regionDoublets, iEvent, iSetup);
}


#endif

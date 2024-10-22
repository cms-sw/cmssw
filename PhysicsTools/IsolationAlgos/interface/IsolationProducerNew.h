#ifndef IsolationAlgos_IsolationProducer_h
#define IsolationAlgos_IsolationProducer_h
/* \class IsolationProducer<C1, C2, Algo>
 *
 * \author Francesco Fabozzi, INFN
 *
 * template class to store isolation
 *
 */
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "CommonTools/UtilAlgos/interface/MasterCollectionHelper.h"
#include <vector>

namespace helper {

  template <typename Alg>
  struct NullIsolationAlgorithmSetup {
    static void init(Alg&, const edm::EventSetup&) {}
  };

  template <typename Alg>
  struct IsolationAlgorithmSetup {
    typedef NullIsolationAlgorithmSetup<Alg> type;
  };
}  // namespace helper

namespace reco {
  namespace modulesNew {

    template <typename C1,
              typename C2,
              typename Alg,
              typename OutputCollection = edm::ValueMap<float>,
              typename Setup = typename helper::IsolationAlgorithmSetup<Alg>::type>
    class IsolationProducer : public edm::stream::EDProducer<> {
    public:
      IsolationProducer(const edm::ParameterSet&);
      ~IsolationProducer() override;

    private:
      void produce(edm::Event&, const edm::EventSetup&) override;
      edm::EDGetTokenT<C1> srcToken_;
      edm::EDGetTokenT<C2> elementsToken_;
      Alg alg_;
    };

    template <typename C1, typename C2, typename Alg, typename OutputCollection, typename Setup>
    IsolationProducer<C1, C2, Alg, OutputCollection, Setup>::IsolationProducer(const edm::ParameterSet& cfg)
        : srcToken_(consumes<C1>(cfg.template getParameter<edm::InputTag>("src"))),
          elementsToken_(consumes<C2>(cfg.template getParameter<edm::InputTag>("elements"))),
          alg_(reco::modules::make<Alg>(cfg)) {
      produces<OutputCollection>();
    }

    template <typename C1, typename C2, typename Alg, typename OutputCollection, typename Setup>
    IsolationProducer<C1, C2, Alg, OutputCollection, Setup>::~IsolationProducer() {}

    template <typename C1, typename C2, typename Alg, typename OutputCollection, typename Setup>
    void IsolationProducer<C1, C2, Alg, OutputCollection, Setup>::produce(edm::Event& evt, const edm::EventSetup& es) {
      using namespace edm;
      using namespace std;
      Handle<C1> src;
      Handle<C2> elements;
      evt.getByToken(srcToken_, src);
      evt.getByToken(elementsToken_, elements);

      Setup::init(alg_, es);

      ::helper::MasterCollection<C1> master(src, evt);
      auto isolations = std::make_unique<OutputCollection>();
      if (!src->empty()) {
        typename OutputCollection::Filler filler(*isolations);
        vector<double> iso(master.size(), -1);
        size_t i = 0;
        for (typename C1::const_iterator lep = src->begin(); lep != src->end(); ++lep)
          iso[master.index(i++)] = alg_(*lep, *elements);
        filler.insert(master.get(), iso.begin(), iso.end());
        filler.fill();
      }
      evt.put(std::move(isolations));
    }

  }  // namespace modulesNew
}  // namespace reco

#endif

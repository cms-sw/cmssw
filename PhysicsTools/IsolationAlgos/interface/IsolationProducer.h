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
#include "DataFormats/Common/interface/AssociationVector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include <vector>

namespace helper {

  template <typename Alg>
  struct NullIsolationAlgorithmSetup {
    using ESConsumesToken = int;
    static ESConsumesToken esConsumes(edm::ConsumesCollector) { return {}; }
    static void init(Alg&, const edm::EventSetup&, ESConsumesToken) {}
  };

  template <typename Alg>
  struct IsolationAlgorithmSetup {
    typedef NullIsolationAlgorithmSetup<Alg> type;
  };
}  // namespace helper

template <typename C1,
          typename C2,
          typename Alg,
          typename OutputCollection = edm::AssociationVector<edm::RefProd<C1>, std::vector<typename Alg::value_type> >,
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
  typename Setup::ESConsumesToken esToken_;
};

template <typename C1, typename C2, typename Alg, typename OutputCollection, typename Setup>
IsolationProducer<C1, C2, Alg, OutputCollection, Setup>::IsolationProducer(const edm::ParameterSet& cfg)
    : srcToken_(consumes<C1>(cfg.template getParameter<edm::InputTag>("src"))),
      elementsToken_(consumes<C2>(cfg.template getParameter<edm::InputTag>("elements"))),
      alg_(reco::modules::make<Alg>(cfg)),
      esToken_(Setup::esConsumes(consumesCollector())) {
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

  Setup::init(alg_, es, esToken_);

  typename OutputCollection::refprod_type ref(src);
  auto isolations = std::make_unique<OutputCollection>(ref);

  size_t i = 0;
  for (typename C1::const_iterator lep = src->begin(); lep != src->end(); ++lep) {
    typename Alg::value_type iso = alg_(*lep, *elements);
    isolations->setValue(i++, iso);
  }
  evt.put(std::move(isolations));
}

#endif

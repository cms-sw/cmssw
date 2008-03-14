#ifndef RecoAlgos_ConstrainedFitCandProducer_h
#define RecoAlgos_ConstrainedFitCandProducer_h
/* \class ConstrainedFitProducer
 *
 * \author Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "PhysicsTools/UtilAlgos/interface/ParameterAdapter.h"
#include "PhysicsTools/UtilAlgos/interface/EventSetupInitTrait.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include <vector>

template<typename Fitter,
	 typename InputCollection = reco::CandidateCollection,
	 typename OutputCollection = InputCollection,
	 typename Init = typename ::reco::modules::EventSetupInit<Fitter>::type>
class ConstrainedFitCandProducer : public edm::EDProducer {
public:
  explicit ConstrainedFitCandProducer(const edm::ParameterSet &);

private:
  edm::InputTag src_;
  Fitter fitter_;
  void produce(edm::Event &, const edm::EventSetup &);
};

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"
#include "FWCore/Framework/interface/EventSetup.h"

template<typename Fitter, typename InputCollection, typename OutputCollection, typename Init>
ConstrainedFitCandProducer<Fitter, InputCollection, OutputCollection, Init>::ConstrainedFitCandProducer(const edm::ParameterSet & cfg) :
  src_(cfg.template getParameter<edm::InputTag>("src")),
  fitter_(reco::modules::make<Fitter>(cfg)) {
  produces<OutputCollection>();
  std::string alias( cfg.getParameter<std::string>("@module_label"));
}

namespace reco {
  namespace fitHelper {
    template<typename C>
    struct Adder {
      static void add(std::auto_ptr<C> & c, std::auto_ptr<reco::VertexCompositeCandidate> t) { c->push_back(*t); }
    };

    template<typename T>
    struct Adder<edm::OwnVector<T> > {
      static void add(std::auto_ptr<edm::OwnVector<T> > & c, std::auto_ptr<reco::VertexCompositeCandidate> t) { c->push_back(t); }
    };

    template<typename C>
      inline void add(std::auto_ptr<C> & c, std::auto_ptr<reco::VertexCompositeCandidate> t) {
      Adder<C>::add(c, t);
    }
  }
}

template<typename Fitter, typename InputCollection, typename OutputCollection, typename Init>
void ConstrainedFitCandProducer<Fitter, InputCollection, OutputCollection, Init>::produce(edm::Event & evt, const edm::EventSetup & es) {
  using namespace edm; 
  using namespace reco;
  using namespace std;
  Init::init(fitter_, es);
  Handle<InputCollection> cands;
  evt.getByLabel(src_, cands);
  auto_ptr<OutputCollection> fitted(new OutputCollection);
  fitted->reserve(cands->size());
  for(typename InputCollection::const_iterator c = cands->begin(); c != cands->end(); ++ c) {
    std::auto_ptr<VertexCompositeCandidate> clone(new VertexCompositeCandidate(*c));
    fitter_.set(*clone);
    fitHelper::add(fitted, clone);
  }
  evt.put(fitted);
}

#endif

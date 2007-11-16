#ifndef RecoAlgos_ConstrainedFitCandProducer_h
#define RecoAlgos_ConstrainedFitCandProducer_h
/* \class ConstrainedFitProducer
 *
 * \author Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/RecoCandidate/interface/FitResult.h"
#include "PhysicsTools/UtilAlgos/interface/ParameterAdapter.h"
#include "PhysicsTools/UtilAlgos/interface/EventSetupInitTrait.h"
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
  bool saveFitResults_;
  Fitter fitter_;
  void produce(edm::Event &, const edm::EventSetup &);
};

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "PhysicsTools/UtilAlgos/interface/MasterCollectionHelper.h"

template<typename Fitter, typename InputCollection, typename OutputCollection, typename Init>
ConstrainedFitCandProducer<Fitter, InputCollection, OutputCollection, Init>::ConstrainedFitCandProducer(const edm::ParameterSet & cfg) :
  src_(cfg.template getParameter<edm::InputTag>("src")),
  saveFitResults_(cfg.template getParameter<bool>("saveFitResults")),
  fitter_(reco::modules::make<Fitter>(cfg)) {
  produces<OutputCollection>();
  std::string alias( cfg.getParameter<std::string>("@module_label"));
  if (saveFitResults_)
    produces<reco::FitResultCollection>().setBranchAlias(alias + "FitResults");
}

namespace reco {
  namespace helper {
    template<typename C>
    struct ValueGetter {
      typedef typename C::value_type value_type;
      static const value_type & get(std::auto_ptr<value_type> t) { return *t; }
    };

    template<typename T>
    struct ValueGetter<edm::OwnVector<T> > {
      static std::auto_ptr<T> get(std::auto_ptr<T> t) { return t; }
    };
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
  FitQuality fq;
  auto_ptr<OutputCollection> refitted(new OutputCollection);
  auto_ptr<FitResultCollection> fitResults(new FitResultCollection);
  FitResultCollection::Filler filler(*fitResults);
  size_t i = 0;
  ::helper::MasterCollection<InputCollection> master(cands);
  vector<FitQuality> q(master.size(), FitQuality());
  for(typename InputCollection::const_iterator c = cands->begin(); c != cands->end(); ++ c) {
    std::auto_ptr<typename InputCollection::value_type> clone(c->clone());
    fq = fitter_.set(*clone);
    refitted->push_back(reco::helper::ValueGetter<InputCollection>::get(clone));
    if (saveFitResults_) q[master.index(i++)];
  }
  evt.put(refitted);
  if (saveFitResults_) { 
    filler.insert(master.get(), q.begin(), q.end());
    filler.fill();
    evt.put(fitResults); 
  }
}

#endif

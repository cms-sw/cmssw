#ifndef RecoAlgos_ConstrainedFitCandProducer_h
#define RecoAlgos_ConstrainedFitCandProducer_h
/* \class ConstrainedFitCandProducer
 *
 * \author Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "CommonTools/UtilAlgos/interface/EventSetupInitTrait.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "SimGeneral/HepPDTRecord/interface/PdtEntry.h"
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
  bool setLongLived_;
  bool setMassConstraint_;
  bool setPdgId_;
  int pdgId_;  
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
#include <algorithm>

template<typename Fitter, typename InputCollection, typename OutputCollection, typename Init>
ConstrainedFitCandProducer<Fitter, InputCollection, OutputCollection, Init>::ConstrainedFitCandProducer(const edm::ParameterSet & cfg) :
  src_(cfg.template getParameter<edm::InputTag>("src")),
  setLongLived_(false), setMassConstraint_(false), setPdgId_(false),
  fitter_(reco::modules::make<Fitter>(cfg)) {
  produces<OutputCollection>();
  std::string alias( cfg.getParameter<std::string>("@module_label"));
  const std::string setLongLived("setLongLived");
  std::vector<std::string> vBoolParams = cfg.template getParameterNamesForType<bool>();
  bool found = find(vBoolParams.begin(), vBoolParams.end(), setLongLived) != vBoolParams.end();
  if(found) setLongLived_ = cfg.template getParameter<bool>("setLongLived");
  const std::string setMassConstraint("setMassConstraint");
  found = find(vBoolParams.begin(), vBoolParams.end(), setMassConstraint) != vBoolParams.end();
  if(found) setMassConstraint_ = cfg.template getParameter<bool>("setMassConstraint");
  const std::string setPdgId("setPdgId");
  std::vector<std::string> vIntParams = cfg.getParameterNamesForType<int>();
  found = find(vIntParams.begin(), vIntParams.end(), setPdgId) != vIntParams.end();
  if(found) { setPdgId_ = true; pdgId_ = cfg.getParameter<int>("setPdgId"); }
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
  Init::init(fitter_, evt, es);
  edm::Handle<InputCollection> cands;
  evt.getByLabel(src_, cands);
  std::auto_ptr<OutputCollection> fitted(new OutputCollection);
  fitted->reserve(cands->size());
  for(typename InputCollection::const_iterator c = cands->begin(); c != cands->end(); ++ c) {
    std::auto_ptr<reco::VertexCompositeCandidate> clone(new reco::VertexCompositeCandidate(*c));
    fitter_.set(*clone);
    if(setLongLived_) clone->setLongLived();
    if(setMassConstraint_) clone->setMassConstraint();
    if(setPdgId_) clone->setPdgId(pdgId_);
    reco::fitHelper::add(fitted, clone);
  }
  evt.put(fitted);
}

#endif

/*
 * Produce a RefVector of PFTaus that pass a given selection on
 * the input discriminator collection.
 *
 * Parameters:
 *  InputTag src = View of PFTaus
 *  InputTag disc = discriinator
 *  double cut
 *
 * Author: Evan K. Friis (UC Davis)
 *
 */

#include <boost/foreach.hpp>
#include <memory>

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"
#include "RecoTauTag/RecoTau/interface/RecoTauCommonUtilities.h"

class ConcreteTauBuilder {
  public:
    typedef std::vector<reco::PFTau> OutputType;
    static OutputType::value_type make(const reco::PFTauRef& ref) {
      return *ref;
    }
};

class RefToBaseBuilder {
  public:
    typedef edm::RefToBaseVector<reco::PFTau> OutputType;
    static OutputType::value_type make(const reco::PFTauRef& ref) {
      return edm::RefToBase<reco::PFTau>(ref);
    }
};

class RefVectorBuilder {
  public:
    typedef reco::PFTauRefVector OutputType;
    static OutputType::value_type make(const reco::PFTauRef& ref) {
      return ref;
    }
};


template<typename T>
class RecoTauDiscriminatorRefSelectorImpl : public edm::EDFilter {
  public:
    explicit RecoTauDiscriminatorRefSelectorImpl(const edm::ParameterSet &pset);
    ~RecoTauDiscriminatorRefSelectorImpl() {}
    bool filter(edm::Event &evt, const edm::EventSetup &es);
  private:
    typedef typename T::OutputType OutputType;
    edm::InputTag src_;
    edm::InputTag discriminatorSrc_;
    double cut_;
    bool filter_;
};

template<typename T>
RecoTauDiscriminatorRefSelectorImpl<T>::RecoTauDiscriminatorRefSelectorImpl(
    const edm::ParameterSet &pset) {
  src_ = pset.getParameter<edm::InputTag>("src");
  discriminatorSrc_ = pset.getParameter<edm::InputTag>("discriminator");
  cut_ = pset.getParameter<double>("cut");
  filter_ = pset.getParameter<bool>("filter");
  //produces<reco::PFTauRefVector>();
  produces<OutputType>();
}


template<typename T>
bool RecoTauDiscriminatorRefSelectorImpl<T>::filter(edm::Event& evt,
                                           const edm::EventSetup &es) {
  edm::Handle<reco::CandidateView> input;
  evt.getByLabel(src_, input);
  reco::PFTauRefVector inputRefs =
      reco::tau::castView<reco::PFTauRefVector>(input);

  edm::Handle<reco::PFTauDiscriminator> disc;
  evt.getByLabel(discriminatorSrc_, disc);

//  std::auto_ptr<reco::PFTauRefVector> output(
//      new reco::PFTauRefVector(inputRefs.id()));
  //std::auto_ptr<OutputType> output(
  //    new OutputType(inputRefs.id()));
  std::auto_ptr<OutputType> output(new OutputType);

  BOOST_FOREACH(reco::PFTauRef ref, inputRefs) {
    if ( (*disc)[ref] > cut_ )
      output->push_back(T::make(ref));
  }
  size_t selected = output->size();
  evt.put(output);
  return (!filter_ || selected);
}

typedef RecoTauDiscriminatorRefSelectorImpl<RefVectorBuilder> RecoTauDiscriminatorRefSelector;
typedef RecoTauDiscriminatorRefSelectorImpl<ConcreteTauBuilder> RecoTauDiscriminatorSelector;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RecoTauDiscriminatorRefSelector);
DEFINE_FWK_MODULE(RecoTauDiscriminatorSelector);

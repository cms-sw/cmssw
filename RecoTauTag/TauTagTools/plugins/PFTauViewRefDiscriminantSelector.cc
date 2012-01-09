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

class RecoTauDiscriminatorRefSelector : public edm::EDFilter {
  public:
    explicit RecoTauDiscriminatorRefSelector(const edm::ParameterSet &pset);
    ~RecoTauDiscriminatorRefSelector() {}
    bool filter(edm::Event &evt, const edm::EventSetup &es);
  private:
    typedef edm::RefToBaseVector<reco::PFTau> OutputType;
    edm::InputTag src_;
    edm::InputTag discriminatorSrc_;
    double cut_;
    bool filter_;
};

RecoTauDiscriminatorRefSelector::RecoTauDiscriminatorRefSelector(
    const edm::ParameterSet &pset) {
  src_ = pset.getParameter<edm::InputTag>("src");
  discriminatorSrc_ = pset.getParameter<edm::InputTag>("discriminator");
  cut_ = pset.getParameter<double>("cut");
  filter_ = pset.getParameter<bool>("filter");
  //produces<reco::PFTauRefVector>();
  produces<OutputType>();
}


bool RecoTauDiscriminatorRefSelector::filter(edm::Event& evt,
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
      output->push_back(edm::RefToBase<reco::PFTau>(ref));
  }
  size_t selected = output->size();
  evt.put(output);
  return (!filter_ || selected);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RecoTauDiscriminatorRefSelector);

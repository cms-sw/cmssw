/*
 * AssociationMatchRefSelector
 *
 * EDFilter template to select objects from a View<InputType> that have valid
 * matches in the given Association<MatchedType> matching, and produce a
 * RefToBaseVector pointing to the matched objects.
 *
 * Author: Evan K. Friis (UC Davis)
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/RefToBaseVector.h"

namespace reco { namespace tau {

template<typename InputType, typename MatchedType>
class AssociationMatchRefSelector : public edm::EDFilter {
  public:
    typedef typename edm::RefToBaseVector<InputType> OutputType;
    typedef typename edm::Association<MatchedType> AssocType;
    explicit AssociationMatchRefSelector(const edm::ParameterSet &pset) {
      src_ = pset.getParameter<edm::InputTag>("src");
      matching_ = pset.getParameter<edm::InputTag>("matching");
      filter_ = pset.getParameter<bool>("filter");
      produces<OutputType>();
    }
    bool filter(edm::Event& evt, const edm::EventSetup &es) {
      edm::Handle<edm::View<InputType> > input;
      evt.getByLabel(src_, input);
      edm::Handle<AssocType> match;
      evt.getByLabel(matching_, match);
      std::auto_ptr<OutputType> output(new OutputType);
      for (size_t i = 0; i < input->size(); ++i) {
        typename AssocType::reference_type matched = (*match)[input->refAt(i)];
        // Check if matched
        if (matched.isNonnull()) {
          output->push_back(input->refAt(i));
        }
      }
      bool notEmpty = output->size();
      evt.put(output);
      // Filter if no events passed
      return ( !filter_ || notEmpty );
    }
  private:
    edm::InputTag src_;
    edm::InputTag matching_;
    bool filter_;
};
}}  // end reco::tau

#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/TauReco/interface/PFTau.h"

typedef reco::tau::AssociationMatchRefSelector<reco::Candidate,
          reco::GenJetCollection>  CandViewGenJetMatchRefSelector;


DEFINE_FWK_MODULE(CandViewGenJetMatchRefSelector);

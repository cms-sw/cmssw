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

namespace {

// When the output is a RefToBaseVector<Candidate>, we need to copy the input
// RefToBases.  (otherwise we get a complaint since Ref<Candidate> is not the
// concrete type.  When storing RefVectors to PFTaus we need to cast the refs
// correctly.

class RefCaster {
  public:
    template<typename InBaseRef, typename REF>
      REF convert(const InBaseRef& in) const {
      return in.template castTo<REF>();
    }
};

class RefCopier {
  public:
    template<typename InBaseRef, typename REF>
      REF convert(const InBaseRef& in) const {
      return REF(in);
    }
};

}

template<typename InputType, typename MatchedType,
  typename OutputType=typename edm::RefToBaseVector<InputType>,
  typename ClonePolicy=RefCopier>
class AssociationMatchRefSelector : public edm::EDFilter {
  public:
    //typedef typename edm::RefToBaseVector<InputType> OutputType;
    typedef typename OutputType::value_type OutputValue;
    typedef typename edm::Association<MatchedType> AssocType;
    typedef typename edm::RefToBase<InputType> InputRef;

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
          OutputValue toPut =
            cloner_.template convert<InputRef, OutputValue>(input->refAt(i));
          output->push_back(toPut);
        }
      }
      bool notEmpty = output->size();
      evt.put(output);
      // Filter if no events passed
      return ( !filter_ || notEmpty );
    }
  private:
    ClonePolicy cloner_;
    edm::InputTag src_;
    edm::InputTag matching_;
    bool filter_;
};
}}  // end reco::tau

#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"

typedef reco::tau::AssociationMatchRefSelector<reco::Candidate,
          reco::GenJetCollection>  CandViewGenJetMatchRefSelector;

typedef reco::tau::AssociationMatchRefSelector<reco::Candidate,
          reco::PFTauCollection, reco::PFTauRefVector,
          reco::tau::RefCaster>  CandViewPFTauMatchRefSelector;

DEFINE_FWK_MODULE(CandViewGenJetMatchRefSelector);
DEFINE_FWK_MODULE(CandViewPFTauMatchRefSelector);

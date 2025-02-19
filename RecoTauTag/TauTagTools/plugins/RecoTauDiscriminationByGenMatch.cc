/*
 * Build a PFTauDiscriminator that returns 1.0 if the a given tau has a matching
 * in the input matching collection.
 *
 * Author: Evan K. Friis (UC Davis)
 *
 */

#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"

class RecoTauDiscriminationByGenMatch : public PFTauDiscriminationProducerBase  {
   public:
      explicit RecoTauDiscriminationByGenMatch(const edm::ParameterSet& pset)
        :PFTauDiscriminationProducerBase(pset){
         matchingSrc_        = pset.getParameter<edm::InputTag>("match");
      }
      ~RecoTauDiscriminationByGenMatch(){}
      double discriminate(const reco::PFTauRef& pfTau);
      virtual void beginEvent(
          const edm::Event& evt, const edm::EventSetup& es);
   private:
      edm::InputTag matchingSrc_;
      edm::Handle<edm::Association<reco::GenJetCollection> > matching_;
};

void RecoTauDiscriminationByGenMatch::beginEvent(
    const edm::Event& evt, const edm::EventSetup& es) {
  evt.getByLabel(matchingSrc_, matching_);
}

double
RecoTauDiscriminationByGenMatch::discriminate(const reco::PFTauRef& tau) {
  reco::GenJetRef genJet = (*matching_)[tau];
  return genJet.isNonnull() ? 1.0 : 0;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RecoTauDiscriminationByGenMatch);

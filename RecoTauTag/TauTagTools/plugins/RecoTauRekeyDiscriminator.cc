/*
 * RecoTauRekeyDiscriminator
 *
 * Author: Evan K. Friis, UC Davis
 *
 * Re-keys the results in a PFTauDiscriminator from one tau collection to
 * another.  Useful in case you make a copy of a tau collection and don't want
 * to recompute the discriminants.
 *
 * Takes as input:
 *
 * PFTauProducer : tau collection to copy the discriminators to
 *
 * matching : matching between old and new tau collections.  Built by the
 * PFTauMatcher module.
 *
 * otherDiscriminator : the original discriminator on the old taus to copy to
 * this tau collection.
 */

#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"

class RecoTauRekeyDiscriminator : public PFTauDiscriminationProducerBase {
  public:
    RecoTauRekeyDiscriminator(const edm::ParameterSet& pset);
    virtual ~RecoTauRekeyDiscriminator() {}
    double discriminate(const reco::PFTauRef& tau);
    void beginEvent(const edm::Event& evt, const edm::EventSetup& evtSetup);
  private:
    typedef edm::Association<reco::PFTauCollection> Matching;
    edm::InputTag otherDiscriminatorSrc_;
    edm::Handle<reco::PFTauDiscriminator> otherDiscriminator_;
    edm::InputTag matchingSrc_;
    edm::Handle<Matching> matching_;
    std::string moduleName_;
    bool verbose_;
};

RecoTauRekeyDiscriminator::RecoTauRekeyDiscriminator(
    const edm::ParameterSet& pset):PFTauDiscriminationProducerBase(pset) {
  otherDiscriminatorSrc_ = pset.getParameter<edm::InputTag>(
      "otherDiscriminator");
  matchingSrc_ = pset.getParameter<edm::InputTag>(
      "matching");
  verbose_ = pset.getUntrackedParameter<bool>("verbose", false);
  moduleName_ = pset.getParameter<std::string>("@module_label");
}

double
RecoTauRekeyDiscriminator::discriminate(const reco::PFTauRef& tau) {
  reco::PFTauRef originalTau = (*matching_)[tau];
  if (originalTau.isNull()) {
    if (verbose_) {
      std::cout << "RecoTauRekey::" << moduleName_
        << "Could not find original PFTau!" << std::endl;
    }
    return prediscriminantFailValue_;
  } else {
    double result = (*otherDiscriminator_)[originalTau];
    if (verbose_)
      std::cout << "RecoTauRekey::" << moduleName_
        << " rekeyed with value: " << result << std::endl;
    return result;
  }
}

void
RecoTauRekeyDiscriminator::beginEvent(
    const edm::Event& evt, const edm::EventSetup& evtSetup) {
  evt.getByLabel(otherDiscriminatorSrc_, otherDiscriminator_);
  evt.getByLabel(matchingSrc_, matching_);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RecoTauRekeyDiscriminator);

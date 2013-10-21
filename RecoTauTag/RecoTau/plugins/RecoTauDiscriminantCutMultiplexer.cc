/*
 * RecoTauDiscriminantCutMultiplexer
 *
 * Author: Evan K. Friis, UW
 *
 * Takes two PFTauDiscriminators.
 *
 * The "key" discriminantor is rounded to the nearest integer.
 *
 * A set of cuts for different keys on the "toMultiplex" discriminantor is
 * provided in the config file.
 *
 * Both the key and toMultiplex discriminators should map to the same PFTau
 * collection.
 *
 */
#include <boost/foreach.hpp>
#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "TMath.h"

class RecoTauDiscriminantCutMultiplexer : public PFTauDiscriminationProducerBase {
  public:
    explicit RecoTauDiscriminantCutMultiplexer(const edm::ParameterSet& pset);

    ~RecoTauDiscriminantCutMultiplexer() {}
    double discriminate(const reco::PFTauRef&);
    void beginEvent(const edm::Event& event, const edm::EventSetup& eventSetup);

  private:
    typedef std::map<int, double> DiscriminantCutMap;

    DiscriminantCutMap cuts_;
    edm::InputTag toMultiplex_;
    edm::InputTag key_;
    edm::Handle<reco::PFTauDiscriminator> toMultiplexHandle_;
    edm::Handle<reco::PFTauDiscriminator> keyHandle_;
};

RecoTauDiscriminantCutMultiplexer::RecoTauDiscriminantCutMultiplexer(
    const edm::ParameterSet& pset):PFTauDiscriminationProducerBase(pset) {
  toMultiplex_ = pset.getParameter<edm::InputTag>("toMultiplex");
  key_ = pset.getParameter<edm::InputTag>("key");
  /*code*/
  typedef std::vector<edm::ParameterSet> VPSet;
  VPSet mapping = pset.getParameter<VPSet>("mapping");
  // Setup our cut map
  BOOST_FOREACH(const edm::ParameterSet &dm, mapping) {
    // Get the mass window for each decay mode
    cuts_.insert(std::make_pair(
          // The category as a key
          dm.getParameter<uint32_t>("category"),
          // The selection
          dm.getParameter<double>("cut")
          ));
  }
}

void RecoTauDiscriminantCutMultiplexer::beginEvent(
    const edm::Event& evt, const edm::EventSetup& es) {
   evt.getByLabel(toMultiplex_, toMultiplexHandle_);
   evt.getByLabel(key_, keyHandle_);
}

double
RecoTauDiscriminantCutMultiplexer::discriminate(const reco::PFTauRef& tau) {
  double disc_result = (*toMultiplexHandle_)[tau];
  double key_result = (*keyHandle_)[tau];
  DiscriminantCutMap::const_iterator cutIter = cuts_.find(TMath::Nint(key_result));

  // Return null if it doesn't exist
  if (cutIter == cuts_.end()) {
    return prediscriminantFailValue_;
  }
  // See if the discriminator passes our cuts
  return disc_result > cutIter->second;
}

DEFINE_FWK_MODULE(RecoTauDiscriminantCutMultiplexer);

#include <boost/foreach.hpp>
#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "FWCore/Utilities/interface/InputTag.h"

class RecoTauDecayModeCutMultiplexer : public PFTauDiscriminationProducerBase {
  public:
    explicit RecoTauDecayModeCutMultiplexer(const edm::ParameterSet& pset);

    ~RecoTauDecayModeCutMultiplexer() {}
    double discriminate(const reco::PFTauRef&);
    void beginEvent(const edm::Event& event, const edm::EventSetup& eventSetup);

  private:
    typedef std::pair<unsigned int, unsigned int> IntPair;
    typedef std::map<IntPair, double> DecayModeCutMap;

    DecayModeCutMap decayModeCuts_;
    edm::InputTag toMultiplex_;
    edm::Handle<reco::PFTauDiscriminator> handle_;
};

RecoTauDecayModeCutMultiplexer::RecoTauDecayModeCutMultiplexer(
    const edm::ParameterSet& pset):PFTauDiscriminationProducerBase(pset) {
  toMultiplex_ = pset.getParameter<edm::InputTag>("toMultiplex");
  typedef std::vector<edm::ParameterSet> VPSet;
  const VPSet& decayModes = pset.getParameter<VPSet>("decayModes");
  // Setup our cut map
  BOOST_FOREACH(const edm::ParameterSet &dm, decayModes) {
    // Get the mass window for each decay mode
    decayModeCuts_.insert(std::make_pair(
            // The decay mode as a key
            std::make_pair(
                dm.getParameter<uint32_t>("nCharged"),
                dm.getParameter<uint32_t>("nPiZeros")),
            // The selection
            dm.getParameter<double>("cut")
        ));
  }
}

void
RecoTauDecayModeCutMultiplexer::beginEvent(
    const edm::Event& evt, const edm::EventSetup& es) {
   evt.getByLabel(toMultiplex_, handle_);
}

double
RecoTauDecayModeCutMultiplexer::discriminate(const reco::PFTauRef& tau) {
  double disc_result = (*handle_)[tau];
  DecayModeCutMap::const_iterator cutIter =
      decayModeCuts_.find(std::make_pair(tau->signalPFChargedHadrCands().size(),
                                         tau->signalPiZeroCandidates().size()));

  // Return null if it doesn't exist
  if (cutIter == decayModeCuts_.end()) {
    return prediscriminantFailValue_;
  }
  // See if the discriminator passes our cuts
  return disc_result > cutIter->second;
}

DEFINE_FWK_MODULE(RecoTauDecayModeCutMultiplexer);

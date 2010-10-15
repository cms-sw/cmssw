#include <boost/foreach.hpp>

#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "CommonTools/Utils/interface/StringObjectFunction.h"
#include "DataFormats/Math/interface/deltaR.h"

class PFRecoTauDiscriminationByHPSSelection
  : public PFTauDiscriminationProducerBase {
  public:
    explicit PFRecoTauDiscriminationByHPSSelection(
        const edm::ParameterSet& pset);

    ~PFRecoTauDiscriminationByHPSSelection() {}
    double discriminate(const reco::PFTauRef&);

  private:
    typedef StringObjectFunction<reco::PFTau> TauFunc;
    typedef std::pair<unsigned int, unsigned int> IntPair;
    typedef std::pair<double, double> DoublePair;
    typedef std::map<IntPair, DoublePair> DecayModeCutMap;

    TauFunc signalConeFun_;
    DecayModeCutMap decayModeCuts_;
    double matchingCone_;
    double minPt_;
};

PFRecoTauDiscriminationByHPSSelection::PFRecoTauDiscriminationByHPSSelection(
    const edm::ParameterSet& pset):PFTauDiscriminationProducerBase(pset),
    signalConeFun_(pset.getParameter<std::string>("coneSizeFormula")) {
  // Get the matchign cut
  matchingCone_ = pset.getParameter<double>("matchingCone");
  minPt_ = pset.getParameter<double>("minTauPt");
  // Get the mass cuts for each decay mode
  typedef std::vector<edm::ParameterSet> VPSet;
  const VPSet& decayModes = pset.getParameter<VPSet>("decayModes");
  BOOST_FOREACH(const edm::ParameterSet &dm, decayModes) {
    // Get the mass window for each decay mode
    decayModeCuts_.insert(std::make_pair(
            // The decay mode as a key
            std::make_pair(
                dm.getParameter<uint32_t>("nCharged"),
                dm.getParameter<uint32_t>("nPiZeros")),
            // The mass window
            std::make_pair(
                dm.getParameter<double>("minMass"),
                dm.getParameter<double>("maxMass"))
            )
        );
  }
}

double
PFRecoTauDiscriminationByHPSSelection::discriminate(const reco::PFTauRef& tau) {

  // Check if we pass the min pt
  if (tau->pt() < minPt_)
    return 0.0;

  // See if we select this decay mode
  DecayModeCutMap::const_iterator massWindowIter =
      decayModeCuts_.find(std::make_pair(tau->signalPFChargedHadrCands().size(),
                                         tau->signalPiZeroCandidates().size()));

  // Check if decay mode is supported
  if (massWindowIter == decayModeCuts_.end()) {
    return 0.0;
  }

  const DoublePair &massWindow = massWindowIter->second;

  // Check if tau fails mass cut
  if (tau->mass() > massWindow.second || tau->mass() < massWindow.first) {
    return 0.0;
  }

  // Check if tau passes matching cone cut
  if (deltaR(tau->p4(), tau->jetRef()->p4()) > matchingCone_) {
    return 0.0;
  }

  // Check if tau passes cone cut
  double cone_size = signalConeFun_(*tau);
  // Check if any charged objects fail the signal cone cut
  BOOST_FOREACH(const PFCandidateRef& cand, tau->signalPFChargedHadrCands()) {
    if (deltaR(cand->p4(), tau->p4()) > cone_size)
      return 0.0;
  }
  // Now check the pizeros
  BOOST_FOREACH(const RecoTauPiZero& cand, tau->signalPiZeroCandidates()) {
    if (deltaR(cand.p4(), tau->p4()) > cone_size)
      return 0.0;
  }

  // Otherwise, we pass!
  return 1.0;
}

DEFINE_FWK_MODULE(PFRecoTauDiscriminationByHPSSelection);

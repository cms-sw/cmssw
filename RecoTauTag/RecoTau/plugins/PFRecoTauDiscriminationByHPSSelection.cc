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
    struct DecayModeCuts {
      double minMass_;
      double maxMass_;
      double minPi0Mass_;
      double maxPi0Mass_;
    };

    typedef StringObjectFunction<reco::PFTau> TauFunc;
    typedef std::pair<unsigned int, unsigned int> IntPair;
    typedef std::pair<double, double> DoublePair;
    typedef std::map<IntPair, DecayModeCuts> DecayModeCutMap;

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
    // The mass window(s)
    DecayModeCuts cuts;
    cuts.minMass_ = dm.getParameter<double>("minMass");
    cuts.maxMass_ = dm.getParameter<double>("maxMass");
    if (dm.exists("minPi0Mass")) {
      cuts.minPi0Mass_ = dm.getParameter<double>("minPi0Mass");
      cuts.maxPi0Mass_ = dm.getParameter<double>("maxPi0Mass");
    } else {
      cuts.minPi0Mass_ = -1.0;
      cuts.maxPi0Mass_ = 1e9;
    }
    decayModeCuts_.insert(std::make_pair(
            // The decay mode as a key
            std::make_pair(
                dm.getParameter<uint32_t>("nCharged"),
                dm.getParameter<uint32_t>("nPiZeros")),
            cuts
          ));
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

  const DecayModeCuts &massWindow = massWindowIter->second;

  // Check if tau fails mass cut
  if (tau->mass() > massWindow.maxMass_ || tau->mass() < massWindow.minMass_) {
    return 0.0;
  }

  // Find the total pizero mass
  reco::Candidate::LorentzVector stripsP4;
  BOOST_FOREACH(const reco::RecoTauPiZero& cand, tau->signalPiZeroCandidates()){
    stripsP4 += cand.p4();
  }

  // Check if it fails the pi 0 IM cut
  if (stripsP4.M() > massWindow.maxPi0Mass_ ||
      stripsP4.M() < massWindow.minPi0Mass_) {
    return 0.0;
  }

  // Check if tau passes matching cone cut
  if (deltaR(tau->p4(), tau->jetRef()->p4()) > matchingCone_) {
    return 0.0;
  }

  // Check if tau passes cone cut
  double cone_size = signalConeFun_(*tau);
  // Check if any charged objects fail the signal cone cut
  BOOST_FOREACH(const reco::PFCandidateRef& cand,
                tau->signalPFChargedHadrCands()) {
    if (deltaR(cand->p4(), tau->p4()) > cone_size)
      return 0.0;
  }
  // Now check the pizeros
  BOOST_FOREACH(const reco::RecoTauPiZero& cand,
                tau->signalPiZeroCandidates()) {
    if (deltaR(cand.p4(), tau->p4()) > cone_size)
      return 0.0;
  }

  // Otherwise, we pass!
  return 1.0;
}

DEFINE_FWK_MODULE(PFRecoTauDiscriminationByHPSSelection);

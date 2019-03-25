#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "FWCore/Utilities/interface/InputTag.h"

/* class PFRecoTauDiscriminationByInvMass
 * created : August 30 2010,
 * contributors : Sami Lehti (sami.lehti@cern.ch ; HIP, Helsinki)
 * contributors : Evan Friis (UC Davis)
 * based on H+ tau ID by Lauri Wendland
 */

class PFRecoTauDiscriminationByInvMass: public PFTauDiscriminationProducerBase {
  public:
    explicit PFRecoTauDiscriminationByInvMass(const edm::ParameterSet& pset)
        :PFTauDiscriminationProducerBase(pset) {
      // If select is not set, just return the invariant mass
      cut_ = pset.exists("select");
      if (cut_) {
        const edm::ParameterSet &select = pset.getParameter<edm::ParameterSet>
            ("select");
        // Get default cuts
        min_default_ = select.getParameter<double>("min");
        max_default_ = select.getParameter<double>("max");
        // Get decay mode specific cuts
        std::vector<std::string> decayModeCutNames =
            select.getParameterNamesForType<edm::ParameterSet>();
        for(auto const& dmName : decayModeCutNames) {
          const edm::ParameterSet &dmPSet =
              select.getParameter<edm::ParameterSet>(dmName);
          unsigned int nCharged = dmPSet.getParameter<unsigned int>("charged");
          unsigned int nPiZero = dmPSet.getParameter<unsigned int>("pizeros");
          double minCut = dmPSet.getParameter<double>("min");
          double maxCut = dmPSet.getParameter<double>("max");
          // Add our dm-specific cut to the map
          decayModeCuts_[std::make_pair(nCharged, nPiZero)] =
              std::make_pair(minCut, maxCut);
        }
      }
    }
    ~PFRecoTauDiscriminationByInvMass() override{}
    double discriminate(const reco::PFTauRef&) const override;

  private:
    typedef std::pair<unsigned int, unsigned int> IntPair;
    typedef std::pair<double, double> DoublePair;
    typedef std::map<IntPair, DoublePair> DecayModeCutMap;
    DecayModeCutMap decayModeCuts_;
    double min_default_;
    double max_default_;
    bool cut_;
};

double
PFRecoTauDiscriminationByInvMass::discriminate(const reco::PFTauRef& tau) const {
  double mass = tau->mass();
  if (cut_) {
    unsigned int charged = tau->signalPFChargedHadrCands().size();
    unsigned int pizeros = tau->signalPiZeroCandidates().size();
    DecayModeCutMap::const_iterator specificCut = decayModeCuts_.find(
        std::make_pair(charged, pizeros));
    // Cut does not exist for this decay mode
    if (specificCut == decayModeCuts_.end() )
      return (mass > min_default_ && mass < max_default_);
    else
      return (mass > specificCut->second.first &&
              mass < specificCut->second.second);
  }
  // If we dont' cut, just return the mass
  return mass;
}

DEFINE_FWK_MODULE(PFRecoTauDiscriminationByInvMass);


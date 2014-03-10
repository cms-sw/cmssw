#ifndef HLTMinDPhiMETFilter_h_
#define HLTMinDPhiMETFilter_h_

/** \class  HLTMinDPhiMETFilter
 *
 *  \brief  This rejects events using the minimum delta phi between a jet and MET.
 *  \author Jia Fu Low (Nov 2013)
 *
 *  This code rejects events when a jet is too close to MET. The angle between
 *  the closest jet and MET in the transverse plane is called the min delta phi.
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETFwd.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METFwd.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/JetCollection.h"


namespace edm {
    class ConfigurationDescriptions;
}

// Class declaration
class HLTMinDPhiMETFilter : public HLTFilter {
  public:
    explicit HLTMinDPhiMETFilter(const edm::ParameterSet & iConfig);
    ~HLTMinDPhiMETFilter();
    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
    virtual bool hltFilter(edm::Event & iEvent, const edm::EventSetup & iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;

  private:
    /// Use pt; otherwise, use et.
    bool usePt_;

    //bool excludePFMuons_;  // currently unused

    /// Output trigger type
    int triggerType_;

    /// Consider only n leading-pt (or et) jets, n = maxNJets_
    int maxNJets_;

    /// Minimum pt requirement for jets
    double minPt_;

    /// Maximum (abs) eta requirement for jets
    double maxEta_;

    /// Minium delta phi between a jet and MET
    double minDPhi_;

    /// Input jet, MET collections
    edm::InputTag metLabel_;
    edm::InputTag calometLabel_;  // only used if metLabel_ is empty
    edm::InputTag jetsLabel_;

    edm::EDGetTokenT<reco::METCollection> m_theMETToken;
    edm::EDGetTokenT<reco::CaloMETCollection> m_theCaloMETToken;
    edm::EDGetTokenT<reco::JetView> m_theJetToken;
};

#endif  // HLTMinDPhiMETFilter_h_


#ifndef HLTPFTauPairLeadTrackDzMatchFilter_h
#define HLTPFTauPairLeadTrackDzMatchFilter_h

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"

/** class HLTPFTauPairLeadTrackDzMatchFilter
 * an HLT filter which picks up a PFTauCollection
 * and passes only events with at least one pair of non-overlapping taus with
 * vertices of leading tracks within some dz
 */

class HLTPFTauPairLeadTrackDzMatchFilter : public HLTFilter {

  public:

    explicit HLTPFTauPairLeadTrackDzMatchFilter(const edm::ParameterSet& conf);
    ~HLTPFTauPairLeadTrackDzMatchFilter();
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
    virtual bool hltFilter(edm::Event& ev, const edm::EventSetup& es, trigger::TriggerFilterObjectWithRefs& filterproduct) const override;

  private:

    edm::InputTag tauSrc_;
    edm::EDGetTokenT<reco::PFTauCollection> tauSrcToken_;
    double tauMinPt_;
    double tauMaxEta_;
    double tauMinDR_;
    double tauLeadTrackMaxDZ_;
    int    triggerType_;

};

#endif

#ifndef HLTMuonL2FromL1TPreFilter_h
#define HLTMuonL2FromL1TPreFilter_h

/** \class HLTMuonL2FromL1TPreFilter
 *
 *
 *  This class is an HLTFilter (-> EDFilter) implementing a first
 *  filtering for HLT muons
 *
 *  \author J. Alcaraz
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "HLTrigger/Muon/interface/HLTMuonL2ToL1TMap.h"

namespace edm {
   class ConfigurationDescriptions;
}

class HLTMuonL2FromL1TPreFilter : public HLTFilter {

  public:
    explicit HLTMuonL2FromL1TPreFilter(const edm::ParameterSet&);
    ~HLTMuonL2FromL1TPreFilter();
    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
    virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;

  private:
    /// input tag of the beam spot
    edm::InputTag                    beamSpotTag_ ;
    edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_ ;

    /// input tag of L2 muons
    edm::InputTag                                          candTag_;
    edm::EDGetTokenT<reco::RecoChargedCandidateCollection> candToken_;

    /// input tag of the preceeding L1 filter in the path
    edm::InputTag                                          previousCandTag_;
    edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> previousCandToken_;

    /// input tag of the map from the L2 seed to the sister L2 seeds of cleaned tracks
    edm::InputTag             seedMapTag_;
    edm::EDGetTokenT<SeedMap> seedMapToken_;

    /// minimum number of muons to fire the trigger
    int minN_;

    /// maxEta cut
    double maxEta_;

    /// |eta| bins for minNstations cut
    /// (#bins must match #minNstations cuts and #minNhits cuts)
    std::vector<double> absetaBins_;

    /// minimum number of muon stations used
    std::vector<int> minNstations_;

    /// minimum number of valid muon hits
    std::vector<int> minNhits_;

    /// choose whether to apply cut on number of chambers (DT+CSC)
    bool cutOnChambers_;

    /// minimum number of valid chambers
    std::vector<int> minNchambers_;

    /// cut on impact parameter wrt to the beam spot
    double maxDr_;

    /// cut on impact parameter wrt to the beam spot
    double minDr_;

    /// cut on dz wrt to the beam spot
    double maxDz_;

    /// dxy significance cut
    double min_DxySig_;

    /// pt threshold in GeV
    double minPt_;

    /// pt uncertainty margin (in number of sigmas)
    double nSigmaPt_;
};

#endif //HLTMuonL2FromL1TPreFilter_h

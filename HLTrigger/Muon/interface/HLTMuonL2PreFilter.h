#ifndef HLTMuonL2PreFilter_h
#define HLTMuonL2PreFilter_h

/** \class HLTMuonL2PreFilter
 *
 *  
 *  This class is an HLTFilter (-> EDFilter) implementing a first
 *  filtering for HLT muons
 *
 *  \author J. Alcaraz
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/MuonSeed/interface/L2MuonTrajectorySeedCollection.h"

class HLTMuonL2PreFilter : public HLTFilter {

  public:
    explicit HLTMuonL2PreFilter(const edm::ParameterSet&);
    ~HLTMuonL2PreFilter();
    virtual bool filter(edm::Event&, const edm::EventSetup&);

  private:
    /// checks if a L2 muon was seed by a fired L1 
    bool isTriggeredByLevel1(reco::TrackRef& l2muon, std::vector<l1extra::L1MuonParticleRef>& firedL1muons);

    /// input tag of the beam spot product
    edm::InputTag beamSpotTag_ ;

    /// input tag identifying the product containing muons
    edm::InputTag candTag_;

    /// input tag identifying the product containing refs to muons passing the previous level
    edm::InputTag previousCandTag_;

    /// minimum number of muons to fire the trigger
    int minN_;

    /// maxEta cut
    double maxEta_;

    /// minimum number of valid muon hits
    int minNhits_;

    /// cut on impact parameter wrt to the beam spot
    double maxDr_;

    /// cut on dz wrt to the beam spot
    double maxDz_;

    /// pt threshold in GeV
    double minPt_;

    /// pt uncertainty margin (in number of sigmas)
    double nSigmaPt_;

    /// should we save the input collection ?
    bool saveTag_;
};

#endif //HLTMuonL2PreFilter_h

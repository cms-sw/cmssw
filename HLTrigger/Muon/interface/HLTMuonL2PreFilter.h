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

namespace edm {
   class ConfigurationDescriptions;
}

class HLTMuonL2PreFilter : public HLTFilter {

  public:
    explicit HLTMuonL2PreFilter(const edm::ParameterSet&);
    ~HLTMuonL2PreFilter();
    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
    virtual bool filter(edm::Event&, const edm::EventSetup&);

  private:
    /// input tag of the beam spot
    edm::InputTag beamSpotTag_ ;

    /// input tag of L2 muons
    edm::InputTag candTag_;

    /// input tag of the preceeding L1 filter in the path
    edm::InputTag previousCandTag_;

    /// input tag of the map from the L2 seed to the sister L2 seeds of cleaned tracks
    edm::InputTag seedMapTag_;

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

    /// cut on impact parameter wrt to the beam spot
    double maxDr_;

    /// cut on dz wrt to the beam spot
    double maxDz_;

    /// pt threshold in GeV
    double minPt_;

    /// pt uncertainty margin (in number of sigmas)
    double nSigmaPt_;

    /// should we save the input collection ?
    bool saveTags_;
};

#endif //HLTMuonL2PreFilter_h

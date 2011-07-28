#ifndef HLTMuonL1Filter_h
#define HLTMuonL1Filter_h

/** \class HLTMuonL1Filter
 *
 *  
 *  This class is an HLTFilter (-> EDFilter) implementing a 
 *  filter on L1 GMT input
 *
 *  \author J. Alcaraz
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"

// CSCTF
#include "DataFormats/L1CSCTrackFinder/interface/L1CSCTrackCollection.h"
#include "L1Trigger/CSCTrackFinder/interface/CSCTrackFinderDataTypes.h"
#include "CondFormats/L1TObjects/interface/L1MuTriggerScales.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerScalesRcd.h"

namespace edm {
   class ConfigurationDescriptions;
}

class HLTMuonL1Filter : public HLTFilter {

  public:
    explicit HLTMuonL1Filter(const edm::ParameterSet&);
    ~HLTMuonL1Filter();
    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
    virtual bool filter(edm::Event&, const edm::EventSetup&);

  private:
    /// input tag identifying the product containing muons
    edm::InputTag candTag_;

    /// input tag identifying the product containing refs to muons passing the previous level
    edm::InputTag previousCandTag_;

    /// max Eta cut
    double maxEta_;

    /// pT threshold 
    double minPt_;

    /// Quality codes:
    ///
    /// 0  .. no muon
    /// 1  .. beam halo muon (CSC)
    /// 2  .. very low quality level 1 (e.g. ignore in single and di-muon trigger)
    /// 3  .. very low quality level 2 (e.g. ignore in single muon trigger use in di-muon trigger)
    /// 4  .. very low quality level 3 (e.g. ignore in di-muon trigger, use in single-muon trigger)
    /// 5  .. unmatched RPC
    /// 6  .. unmatched DT or CSC
    /// 7  .. matched DT-RPC or CSC-RPC
    ///
    /// attention: try not to rely on quality codes in analysis: they may change again
    ///
    /// Quality bit mask:
    ///
    /// the eight lowest order or least significant bits correspond to the qulity codes above; 
    /// if a bit is 1, that code is accepted, otherwise not;
    /// example: 11101000 accepts qualities 3, 5, 6, 7
    int qualityBitMask_;

    /// required number of passing candidates to pass the filter
    int minN_;

    /// should we exclude single-segment CSC trigger objects from our counting?
    bool excludeSingleSegmentCSC_;

    /// checks if the passed L1MuExtraParticle is a single segment CSC
    bool isSingleSegmentCSC(const l1extra::L1MuonParticleRef &);

    /// input tag identifying the product containing CSCTF tracks
    edm::InputTag csctfTag_;
    
    /// handle for CSCTFtracks
    edm::Handle<L1CSCTrackCollection> csctfTracks_;

    /// trigger scales
    const L1MuTriggerScales *l1MuTriggerScales_;

    /// trigger scales cache ID
    unsigned long long m_scalesCacheID_ ;

    /// should we save the input collection?
    bool saveTags_;
};

#endif //HLTMuonL1Filter_h

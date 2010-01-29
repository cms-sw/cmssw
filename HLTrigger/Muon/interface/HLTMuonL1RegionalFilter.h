#ifndef HLTMuonL1RegionalFilter_h
#define HLTMuonL1RegionalFilter_h

/** \class HLTMuonL1RegionalFilter
 *
 *  
 *  This filter cuts on MinPt and Quality in specified eta regions
 *
 *  $Date: 2010/01/29 12:15:46 $
 *  $Revision: 1.1 $
 *
 *  \author Cristina Botta, Zoltan Gecse
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/Common/interface/Handle.h"

class HLTMuonL1RegionalFilter : public HLTFilter {

  public:
    explicit HLTMuonL1RegionalFilter(const edm::ParameterSet&);
    ~HLTMuonL1RegionalFilter();
    virtual bool filter(edm::Event&, const edm::EventSetup&);

  private:
    /// prints a formated table of the filter parameters, useful for debugging
    std::string dumpParameters();

    /// prints a formated table of the event, useful for debugging 
    std::string dumpEvent(edm::Handle<l1extra::L1MuonParticleCollection>& allMuons, std::vector<l1extra::L1MuonParticleRef>& prevMuons, trigger::TriggerFilterObjectWithRefs* filterproduct);

    /// input tag identifying the product containing muons
    edm::InputTag candTag;

    /// input tag identifying the product containing refs to muons passing the previous level
    edm::InputTag previousCandTag;

    /// the vector of eta region boundaries; note: # of boundaries = # of regions + 1
    std::vector<double> etaBoundaries;

    /// the vector of MinPt values, one for eta each region
    std::vector<double> minPts;

    /// Quality codes (corresponding byte words):
    ///
    /// 0 (1)  .. no muon
    /// 1 (2)  .. beam halo muon (CSC)
    /// 2 (4)  .. very low quality level 1 (e.g. ignore in single and di-muon trigger)
    /// 3 (8)  .. very low quality level 2 (e.g. ignore in single muon trigger use in di-muon trigger)
    /// 4 (16) .. very low quality level 3 (e.g. ignore in di-muon trigger, use in single-muon trigger)
    /// 5 (32) .. unmatched RPC
    /// 6 (64) .. unmatched DT or CSC
    /// 7 (128).. matched DT-RPC or CSC-RPC
    ///
    /// attention: try not to rely on quality codes in analysis: they may change again
    ///
    /// Quality bit mask:
    ///
    /// a quality mask is the addition of byte words of required qualities. Example: 192 = code 6 or 7 
    /// 
    /// the vector of quality bit masks, one for each eta region
    std::vector<int> qualityBitMasks;

    /// required number of passing candidates to pass the filter
    int minN;

    /// should we save the input collection ?
    bool saveTag;
};

#endif //HLTMuonL1RegionalFilter_h


#ifndef HLTMuonL1RegionalFilter_h
#define HLTMuonL1RegionalFilter_h

/** \class HLTMuonL1RegionalFilter
 *
 *  
 *  This filter cuts on MinPt and Quality in specified eta regions
 *
 *  $Date: 2012/01/21 14:57:04 $
 *  $Revision: 1.4 $
 *
 *  \author Cristina Botta, Zoltan Gecse
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

class HLTMuonL1RegionalFilter : public HLTFilter {

  public:
    explicit HLTMuonL1RegionalFilter(const edm::ParameterSet&);
    ~HLTMuonL1RegionalFilter();
    virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct);

  private:
    /// input tag identifying the product containing muons
    edm::InputTag candTag_;

    /// input tag identifying the product containing refs to muons passing the previous level
    edm::InputTag previousCandTag_;

    /// the vector of eta region boundaries; note: # of boundaries = # of regions + 1
    std::vector<double> etaBoundaries_;

    /// the vector of MinPt values, one for eta each region
    std::vector<double> minPts_;

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
    /// 
    /// the vector of quality bit masks, one for each eta region
    std::vector<int> qualityBitMasks_;

    /// required number of passing candidates to pass the filter
    int minN_;
};

#endif //HLTMuonL1RegionalFilter_h


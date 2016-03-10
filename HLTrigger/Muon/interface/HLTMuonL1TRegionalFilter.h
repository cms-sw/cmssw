#ifndef HLTMuonL1TRegionalFilter_h
#define HLTMuonL1TRegionalFilter_h

/** \class HLTMuonL1TRegionalFilter
 *
 *
 *  This filter cuts on MinPt and Quality in specified eta regions
 *
 *
 *  \author Cristina Botta, Zoltan Gecse
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "DataFormats/L1Trigger/interface/Muon.h"

namespace edm {
  class ConfigurationDescriptions;
}

class HLTMuonL1TRegionalFilter : public HLTFilter {

  public:
    explicit HLTMuonL1TRegionalFilter(const edm::ParameterSet&);
    ~HLTMuonL1TRegionalFilter();
    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
    virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;

  private:
    /// input tag identifying the product containing muons
    edm::InputTag                                       candTag_;
    edm::EDGetTokenT<l1t::MuonBxCollection>             candToken_;

    /// input tag identifying the product containing refs to muons passing the previous level
    edm::InputTag                                          previousCandTag_;
    edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> previousCandToken_;

    /// the vector of eta region boundaries; note: # of boundaries = # of regions + 1
    std::vector<double> etaBoundaries_;

    /// the vector of MinPt values, one for eta each region
    std::vector<double> minPts_;

    /// Quality codes:
    /// to be updated with new L1 quality definitions
    std::vector<int> qualityBitMasks_;

    /// required number of passing candidates to pass the filter
    int minN_;

    /// use central bx only muons
    bool centralBxOnly_;
};

#endif //HLTMuonL1TRegionalFilter_h


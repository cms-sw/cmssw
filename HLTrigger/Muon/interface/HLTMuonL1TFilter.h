#ifndef HLTMuonL1TFilter_h
#define HLTMuonL1TFilter_h

/* \class HLTMuonL1TFilter
 *
 * This is an HLTFilter implementing filtering on L1T Stage2 GMT objects
 * 
 * \author:  V. Rekovic
*/

// user include files

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "DataFormats/L1Trigger/interface/Muon.h"


class HLTMuonL1TFilter : public HLTFilter {

   public:

      explicit HLTMuonL1TFilter(const edm::ParameterSet&);
      ~HLTMuonL1TFilter();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;

   private:

    //input tag identifying the product containing muons
    edm::InputTag                           candTag_;
    edm::EDGetTokenT<l1t::MuonBxCollection> candToken_;

    /// input tag identifying the product containing refs to muons passing the previous level
    edm::InputTag                                          previousCandTag_;
    edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> previousCandToken_;
    
    /// max Eta cut
    double maxEta_;
    
    /// pT threshold
    double minPt_;

    /// Quality codes:
    /// to be updated with new L1 quality definitions
    int qualityBitMask_;
    
    /// min N objects
    double minN_;
    
    /// use central bx only muons
    bool centralBxOnly_;
    
};

#endif //HLTMuonL1TFilter_h

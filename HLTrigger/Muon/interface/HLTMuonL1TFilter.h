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
    
    /// min N objects
    double minN_;
    
    /// use central bx only muons
    bool centralBxOnly_;
    
};

#endif //HLTMuonL1TFilter_h

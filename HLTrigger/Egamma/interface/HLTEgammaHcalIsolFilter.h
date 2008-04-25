#ifndef HLTEgammaHcalIsolFilter_h
#define HLTEgammaHcalIsolFilter_h

/** \class HLTEgammaHcalIsolFilter
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

//
// class decleration
//

class HLTEgammaHcalIsolFilter : public HLTFilter {

   public:
      explicit HLTEgammaHcalIsolFilter(const edm::ParameterSet&);
      ~HLTEgammaHcalIsolFilter();
      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      edm::InputTag candTag_; // input tag identifying product contains filtered photons
      edm::InputTag isoTag_; // input tag identifying product contains hcal isolation map
      edm::InputTag nonIsoTag_; // input tag identifying product contains hcal isolation map
      double hcalisolbarrelcut_;   // Hcal isolation threshold in GeV for barrel 
      double hcalisolendcapcut_;   // Hcal isolation threshold in GeV for endcap
      double HoverEcut_;       // Isolation threshold in H/E
      double HoverEt2cut_;        // Isolation threshold in H/(E**2)
      int    ncandcut_;        // number of photons required
      bool doIsolated_;

      bool   store_;
      edm::InputTag L1IsoCollTag_; 
      edm::InputTag L1NonIsoCollTag_; 
};

#endif //HLTEgammaHcalIsolFilter_h



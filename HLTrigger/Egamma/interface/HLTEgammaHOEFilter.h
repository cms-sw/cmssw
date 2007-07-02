#ifndef HLTEgammaHOEFilter_h
#define HLTEgammaHOEFilter_h

/** \class HLTEgammaHOEFilter
 *
 *  \author Monica Vazquez Acosta (CERN)
 * identical to old HLTEgammaHcalIsolFilter but 
 *  -the Hcal et is devided by the supercluster et
 *  -the eta-range is not restricted to |eta|<2.5
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

//
// class decleration
//

class HLTEgammaHOEFilter : public HLTFilter {

   public:
      explicit HLTEgammaHOEFilter(const edm::ParameterSet&);
      ~HLTEgammaHOEFilter();
      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      edm::InputTag candTag_; // input tag identifying product contains filtered photons
      edm::InputTag isoTag_; // input tag identifying product contains hcal isolation map
      edm::InputTag nonIsoTag_; // input tag identifying product contains hcal isolation map
      double hcalisolbarrelcut_;   // Hcal isolation threshold in GeV for barrel 
      double hcalisolendcapcut_;   // Hcal isolation threshold in GeV for endcap
      int    ncandcut_;        // number of photons required
      bool doIsolated_;
};

#endif //HLTEgammaHOEFilter_h



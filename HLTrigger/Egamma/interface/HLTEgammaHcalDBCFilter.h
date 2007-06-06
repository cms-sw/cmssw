#ifndef HLTEgammaHcalDBCFilter_h
#define HLTEgammaHcalDBCFilter_h

/** \class HLTEgammaHcalDBCFilter
 *
 *  \author Monica Vazquez Acosta (CERN)
 *   Hcal double cone isolation filter
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

//
// class decleration
//

class HLTEgammaHcalDBCFilter : public HLTFilter {

   public:
      explicit HLTEgammaHcalDBCFilter(const edm::ParameterSet&);
      ~HLTEgammaHcalDBCFilter();
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

#endif //HLTEgammaHcalDBCFilter_h



#ifndef HLTPMDocaFilter_h
#define HLTPMDocaFilter_h

/** \class HLTPMDocaFilter
 *
 *  Original Author: Jeremy Werner
 *  Institution: Princeton University, USA                        
 *  Contact: Jeremy.Werner@cern.ch 
 *  Date: February 21, 2007  
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

//
// class decleration
//

class HLTPMDocaFilter : public HLTFilter {

   public:
      explicit HLTPMDocaFilter(const edm::ParameterSet&);
      ~HLTPMDocaFilter();
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct);

   private:
      edm::InputTag candTag_;     // input tag identifying product contains filtered egammas
      double docaDiffPerpCutHigh_;
      double docaDiffPerpCutLow_;
      int    nZcandcut_;           // number of electrons required

};

#endif //HLTPMDocaFilter_h



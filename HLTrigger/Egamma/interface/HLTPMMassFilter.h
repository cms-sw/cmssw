#ifndef HLTPMMassFilter_h
#define HLTPMMassFilter_h

/** \class HLTPMMassFilter
 *
 *  Original Author: Jeremy Werner 
 *  Institution: Princeton University, USA
 *  Contact: Jeremy.Werner@cern.ch
 *  Date: February 21, 2007
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

//
// class declaration
//

class HLTPMMassFilter : public HLTFilter {

   public:
      explicit HLTPMMassFilter(const edm::ParameterSet&);
      ~HLTPMMassFilter();
      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      edm::InputTag candTag_;     // input tag identifying product contains filtered egammas
      edm::InputTag elecTag_;
      double lowerMassCut_;
      double upperMassCut_;
      int    ncandcut_;           // number of electrons required

};

#endif //HLTPMMassFilter_h



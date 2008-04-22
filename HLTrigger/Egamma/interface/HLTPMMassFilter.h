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
      double lowerMassCut_;
      double upperMassCut_;
      int    nZcandcut_;           // number of Z candidates required

      bool   store_;
      bool   relaxed_;
      edm::InputTag L1IsoCollTag_; 
      edm::InputTag L1NonIsoCollTag_; 
 

};

#endif //HLTPMMassFilter_h



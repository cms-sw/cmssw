#ifndef HLTEgammaEtFilter_h
#define HLTEgammaEtFilter_h

/** \class HLTEgammaEtFilter
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

//
// class decleration
//

class HLTEgammaEtFilter : public HLTFilter {

   public:
      explicit HLTEgammaEtFilter(const edm::ParameterSet&);
      ~HLTEgammaEtFilter();
      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      edm::InputTag inputTag_; // input tag identifying product contains egammas
      double etcut_;           // Et threshold in GeV 
      int    ncandcut_;        // number of egammas required
};

#endif //HLTEgammaEtFilter_h

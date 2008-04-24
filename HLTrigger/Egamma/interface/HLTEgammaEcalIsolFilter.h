#ifndef HLTEgammaEcalIsolFilter_h
#define HLTEgammaEcalIsolFilter_h

/** \class HLTEgammaEcalIsolFilter
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

//
// class decleration
//

class HLTEgammaEcalIsolFilter : public HLTFilter {

   public:
      explicit HLTEgammaEcalIsolFilter(const edm::ParameterSet&);
      ~HLTEgammaEcalIsolFilter();
      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      edm::InputTag candTag_; // input tag identifying product contains filtered egammas
      edm::InputTag isoTag_; // input tag identifying product contains ecal isolation map
      edm::InputTag nonIsoTag_; // input tag identifying product contains ecal isolation map
      double ecalisolcut_;   // Ecal isolation threshold in GeV 
      double ecalFracCut_;
      double ecalIsoloEt2_;
      int    ncandcut_;        // number of egammas required
      bool doIsolated_;
};

#endif //HLTEgammaEcalIsolFilter_h



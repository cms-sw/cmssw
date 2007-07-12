#ifndef HLTEcalIsolationFilter_h
#define HLTEcalIsolationFilter_h

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

class HLTEcalIsolationFilter : public HLTFilter {

   public:
      explicit HLTEcalIsolationFilter(const edm::ParameterSet&);
      ~HLTEcalIsolationFilter();
      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      edm::InputTag candTag_; // input tag identifying product contains filtered egammas
      double maxennearby;   // Ecal isolation threshold in GeV 
      double minen;        // number of egammas required
};

#endif 

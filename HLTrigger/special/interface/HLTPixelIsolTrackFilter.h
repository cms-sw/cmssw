#ifndef HLTPixelIsolTrackFilter_h
#define HLTPixelIsolTrackFilter_h

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

class HLTPixelIsolTrackFilter : public HLTFilter {

   public:
      explicit HLTPixelIsolTrackFilter(const edm::ParameterSet&);
      ~HLTPixelIsolTrackFilter();
      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      edm::InputTag candTag_; // input tag identifying product contains filtered egammas
      double maxptnearby;   // Ecal isolation threshold in GeV 
      double minpttrack;        // number of egammas required
};

#endif //HLTEgammaEcalIsolFilter_h

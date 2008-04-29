#ifndef HLTPixelIsolTrackFilter_h
#define HLTPixelIsolTrackFilter_h

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

class HLTPixelIsolTrackFilter : public HLTFilter {

   public:
      explicit HLTPixelIsolTrackFilter(const edm::ParameterSet&);
      ~HLTPixelIsolTrackFilter();
      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      edm::InputTag candTag_; 
      double maxptnearby;    
      double minpttrack;        
      double maxetatrack;
      bool filterE_;
      double minEnergy_;
};

#endif 

#ifndef HLTPixelIsolTrackFilter_h
#define HLTPixelIsolTrackFilter_h

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

class HLTPixelIsolTrackFilter : public HLTFilter {

   public:
      explicit HLTPixelIsolTrackFilter(const edm::ParameterSet&);
      ~HLTPixelIsolTrackFilter();
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct);

   private:
      edm::InputTag candTag_; 
      edm::InputTag hltGTseedlabel_;
      double maxptnearby_;    
      double minpttrack_;        
      double minetatrack_;
      double maxetatrack_;
      bool filterE_;
      double minEnergy_;
      int nMaxTrackCandidates_;
      bool dropMultiL2Event_;
      double minDeltaPtL1Jet_;
};

#endif 

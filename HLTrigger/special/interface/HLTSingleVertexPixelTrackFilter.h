// $Id: HLTSingleVertexPixelTrackFilter.h,v 1.2 2009/11/01 19:35:03 davidlw Exp $

#ifndef HLTSingleVertexPixelTrackFilter_h
#define HLTSingleVertexPixelTrackFilter_h

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

//
// class declaration
//

class HLTSingleVertexPixelTrackFilter : public HLTFilter {

   public:
      explicit HLTSingleVertexPixelTrackFilter(const edm::ParameterSet&);
      ~HLTSingleVertexPixelTrackFilter();
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct);

   private:
      edm::InputTag pixelVerticesTag_;  // input tag identifying product containing Pixel-vertices
      edm::InputTag pixelTracksTag_;  // input tag identifying product containing Pixel-tracks

      double min_Pt_;          // min pt cut
      double max_Pt_;          // max pt cut
      double max_Eta_;          // max eta cut
      double max_Vz_;          // max vz cut
      int min_trks_;  // minimum number of tracks from one vertex
      float min_sep_;          // minimum separation of two tracks in phi-eta
};

#endif //HLTSingleVertexPixelTrackFilter_h

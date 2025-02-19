#ifndef HLTPixlMBForAlignmentFilter_h
#define HLTPixlMBForAlignmentFilter_h

/** \class HLTFiltCand
 *
 *  
 *  This class is an HLTFilter (-> EDFilter) implementing a minimum-bias
 *  HLT trigger acting on candidates, requiring tracks in Pixel det
 *
 *  $Date: 2012/01/21 15:00:14 $
 *
 *  \author Mika Huhtinen
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

//
// class declaration
//

class HLTPixlMBForAlignmentFilter : public HLTFilter {

   public:
      explicit HLTPixlMBForAlignmentFilter(const edm::ParameterSet&);
      ~HLTPixlMBForAlignmentFilter();
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct);

   private:
      edm::InputTag pixlTag_;  // input tag identifying product containing Pixel-tracks

      double min_Pt_;          // min pt cut
      unsigned int min_trks_;  // minimum number of tracks from same vertex
      float min_sep_;          // minimum separation of two tracks in phi-eta
      float min_isol_;         // size of isolation cone around track

};

#endif //HLTPixlMBForAlignmentFilter_h

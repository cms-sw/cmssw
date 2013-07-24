#ifndef HLTPixlMBFilt_h
#define HLTPixlMBFilt_h

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

class HLTPixlMBFilt : public HLTFilter {

   public:
      explicit HLTPixlMBFilt(const edm::ParameterSet&);
      ~HLTPixlMBFilt();
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct);

   private:
      edm::InputTag pixlTag_;  // input tag identifying product containing Pixel-tracks

      double min_Pt_;          // min pt cut
      unsigned int min_trks_;  // minimum number of tracks from one vertex
      float min_sep_;          // minimum separation of two tracks in phi-eta

};

#endif //HLTPixlMBFilt_h

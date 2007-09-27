#ifndef HLTPixlMBSelectFilter_h
#define HLTPixlMBSelectFilter_h

/** \class HLTFiltCand
 *
 *  
 *  This class is an HLTFilter (-> EDFilter) implementing a minimum-bias
 *  HLT trigger acting on candidates, requiring tracks in Pixel det
 *
 *  $Date: 2007/03/30 15:56:10 $
 *
 *  \author Mika Huhtinen
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

//
// class declaration
//

class HLTPixlMBSelectFilter : public HLTFilter {

   public:
      explicit HLTPixlMBSelectFilter(const edm::ParameterSet&);
      ~HLTPixlMBSelectFilter();
      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      edm::InputTag pixlTag_;  // input tag identifying product containing Pixel-tracks

      double min_Pt_;          // min pt cut
      unsigned int min_trks_;  // minimum number of tracks from same vertex
      float min_sep_;          // minimum separation of two tracks in phi-eta
      float min_isol_;         // size of isolation cone around track

};

#endif //HLTPixlMBSelectFilter_h

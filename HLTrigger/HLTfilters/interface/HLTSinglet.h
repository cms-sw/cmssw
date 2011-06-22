#ifndef HLTSinglet_h
#define HLTSinglet_h

/** \class HLTSinglet
 *
 *  
 *  This class is an HLTFilter (-> EDFilter) implementing a basic HLT
 *  trigger for single objects of the same physics type, cutting on
 *  variables relating to their 4-momentum representation
 *
 *  $Date: 2008/05/05 15:48:33 $
 *  $Revision: 1.3 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include<vector>

//
// class declaration
//

template<typename T, int Tid>
class HLTSinglet : public HLTFilter {

   public:

      explicit HLTSinglet(const edm::ParameterSet&);
      ~HLTSinglet();
      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      edm::InputTag inputTag_; // input tag identifying product
      bool saveTags_;           // whether to save this tag
      double min_Pt_;          // pt threshold in GeV 
      double max_Eta_;         // eta range (symmetric)
      int    min_N_;           // number of objects passing cuts required
};

#endif //HLTSinglet_h

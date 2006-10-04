#ifndef HLTSinglet_h
#define HLTSinglet_h

/** \class HLTSinglet
 *
 *  
 *  This class is an HLTFilter (-> EDFilter) implementing a basic HLT
 *  trigger for single objects of the same physics type, cutting on
 *  variables relating to their 4-momentum representation
 *
 *  $Date: 2006/08/14 16:29:11 $
 *  $Revision: 1.12 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include<vector>

//
// class declaration
//

template<typename T>
class HLTSinglet : public HLTFilter {

   public:

      explicit HLTSinglet(const edm::ParameterSet&);
      ~HLTSinglet();
      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      edm::InputTag inputTag_; // input tag identifying product
      double min_Pt_;          // pt threshold in GeV 
      double max_Eta_;         // eta range (symmetric)
      int    min_N_;           // number of objects passing cuts required
};

#endif //HLTSinglet_h

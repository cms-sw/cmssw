#ifndef HLTDoublet_h
#define HLTDoublet_h

/** \class HLTDoublet
 *
 *  
 *  This class is an HLTFilter (-> EDFilter) implementing a basic HLT
 *  trigger for pairs of object, evaluating all pairs with the first
 *  object from collection 1, and the second object from collection 2,
 *  cutting on variables relating to their 4-momentum representations.
 *  The object collections are assumed to be outputs of HLTSinglet
 *  single-object-type filters so that the access is thorugh
 *  RefToBases and polymorphic.
 *
 *  $Date: 2006/08/14 15:26:43 $
 *  $Revision: 1.8 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include<vector>

//
// class decleration
//

class HLTDoublet : public HLTFilter {

   public:

      explicit HLTDoublet(const edm::ParameterSet&);
      ~HLTDoublet();
      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      edm::InputTag inputTag1_;   // input tag identifying 1st product
      edm::InputTag inputTag2_;   // input tag identifying 2nd product
      double Min_Dphi_,Max_Dphi_; // Delta phi window
      double Min_Deta_,Max_Deta_; // Delta eta window
      double Min_Minv_,Max_Minv_; // Minv(1,2) window
      int    Min_N_;              // number of pairs passing cuts required

      bool   same;                // 1st and 2nd product are one and the same
};

#endif //HLTDoublet_h

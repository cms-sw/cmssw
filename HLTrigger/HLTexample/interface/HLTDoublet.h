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
 *  $Date: 2007/04/13 15:57:57 $
 *  $Revision: 1.17 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include<vector>

//
// class declaration
//

class HLTDoublet : public HLTFilter {

   public:

      explicit HLTDoublet(const edm::ParameterSet&);
      ~HLTDoublet();
      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      // configuration
      edm::InputTag inputTag1_;   // input tag identifying 1st product
      edm::InputTag inputTag2_;   // input tag identifying 2nd product
      double min_Dphi_,max_Dphi_; // Delta phi window
      double min_Deta_,max_Deta_; // Delta eta window
      double min_Minv_,max_Minv_; // Minv(1,2) window
      int    min_N_;              // number of pairs passing cuts required

      // calculated from configuration in c'tor
      bool   same_;                      // 1st and 2nd product are one and the same
      bool   cutdphi_,cutdeta_,cutminv_; // cuts are on=true or off=false
};

#endif //HLTDoublet_h

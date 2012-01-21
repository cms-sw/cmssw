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
 *  $Date: 2010/09/26 10:38:10 $
 *  $Revision: 1.4 $
 *
 *  \author Martin Grunewald
 *
 */

#include "DataFormats/Common/interface/Ref.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include<vector>

//
// class declaration
//

template<typename T1, int Tid1, typename T2, int Tid2>
class HLTDoublet : public HLTFilter {

   public:

      explicit HLTDoublet(const edm::ParameterSet&);
      ~HLTDoublet();
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct);

   private:
      // configuration
      edm::InputTag inputTag1_;   // input tag identifying 1st product
      edm::InputTag inputTag2_;   // input tag identifying 2nd product
      double min_Dphi_,max_Dphi_; // Delta phi window
      double min_Deta_,max_Deta_; // Delta eta window
      double min_Minv_,max_Minv_; // Minv(1,2) window
      double min_DelR_,max_DelR_; // Delta R window
      int    min_N_;              // number of pairs passing cuts required

      // calculated from configuration in c'tor
      bool   same_;                      // 1st and 2nd product are one and the same
      bool   cutdphi_,cutdeta_,cutminv_,cutdelr_; // cuts are on=true or off=false

      //
      typedef std::vector<T1> T1Collection;
      typedef edm::Ref<T1Collection> T1Ref;
      std::vector<T1Ref> coll1_;
      typedef std::vector<T2> T2Collection;
      typedef edm::Ref<T2Collection> T2Ref;
      std::vector<T2Ref> coll2_;


};

#endif //HLTDoublet_h

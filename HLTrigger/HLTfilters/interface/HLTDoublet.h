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
 *  $Date: 2012/02/24 13:34:20 $
 *  $Revision: 1.9 $
 *
 *  \author Martin Grunewald
 *
 */

#include "DataFormats/Common/interface/Ref.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include<string>
#include<vector>

//
// class declaration
//

template<typename T1, typename T2>
class HLTDoublet : public HLTFilter {

   public:

      explicit HLTDoublet(const edm::ParameterSet&);
      ~HLTDoublet();
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct);

   private:
      // configuration
      edm::InputTag originTag1_;  // input tag identifying original 1st product
      edm::InputTag originTag2_;  // input tag identifying original 2nd product
      edm::InputTag inputTag1_;   // input tag identifying filtered 1st product
      edm::InputTag inputTag2_;   // input tag identifying filtered 2nd product
      int triggerType1_;
      int triggerType2_;
      double min_Dphi_,max_Dphi_; // Delta phi window
      double min_Deta_,max_Deta_; // Delta eta window
      double min_Minv_,max_Minv_; // Minv(1,2) window
      double min_DelR_,max_DelR_; // Delta R window
      double min_Pt_  ,max_Pt_;   // Pt(1,2) window
      int    min_N_;              // number of pairs passing cuts required

      // calculated from configuration in c'tor
      bool   same_;                      // 1st and 2nd product are one and the same
      bool   cutdphi_,cutdeta_,cutminv_,cutdelr_,cutpt_; // cuts are on=true or off=false

      std::string label_;         // module label

      //
      typedef std::vector<T1> T1Collection;
      typedef edm::Ref<T1Collection> T1Ref;
      std::vector<T1Ref> coll1_;
      typedef std::vector<T2> T2Collection;
      typedef edm::Ref<T2Collection> T2Ref;
      std::vector<T2Ref> coll2_;

};

#endif //HLTDoublet_h

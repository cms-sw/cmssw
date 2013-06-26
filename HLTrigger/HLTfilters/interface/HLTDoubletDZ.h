#ifndef HLTDoubletDZ_h
#define HLTDoubletDZ_h

// 
// Class imlements |dZ|<Max for a pair of two objects
//

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include<string>
#include<vector>

//
// class declaration
//

template<typename T1, typename T2>
class HLTDoubletDZ : public HLTFilter {

   public:

      explicit HLTDoubletDZ(const edm::ParameterSet&);
      ~HLTDoubletDZ();
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
      double minDR_;              // minimum dR between two objects to be considered a pair
      double maxDZ_;              // number of pairs passing cuts required
      bool   same_;               // 1st and 2nd product are one and the same
      int    min_N_;              // number of pairs passing cuts required
      bool   checkSC_;            // make sure SC constituents are different

      std:: string label_;        // module label

      typedef std::vector<T1> T1Collection;
      typedef edm::Ref<T1Collection> T1Ref;
      std::vector<T1Ref> coll1_;
      typedef std::vector<T2> T2Collection;
      typedef edm::Ref<T2Collection> T2Ref;
      std::vector<T2Ref> coll2_;


};

#endif //HLTDoubletDZ_h

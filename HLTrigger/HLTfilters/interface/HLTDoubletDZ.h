#ifndef HLTDoubletDZ_h
#define HLTDoubletDZ_h

// 
// Class imlements |dZ|<Max for a pair of two objects
//

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include<vector>

//
// class declaration
//

template<typename T1, int Tid1, typename T2, int Tid2>
class HLTDoubletDZ : public HLTFilter {

   public:

      explicit HLTDoubletDZ(const edm::ParameterSet&);
      ~HLTDoubletDZ();
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct);

   private:
      // configuration
      edm::InputTag inputTag1_;   // input tag identifying 1st product
      edm::InputTag inputTag2_;   // input tag identifying 2nd product
      double minDR_;              // minimum dR between two objects to be considered a pair
      double maxDZ_;              // number of pairs passing cuts required
      bool   same_;               // 1st and 2nd product are one and the same
      int    min_N_;              // number of pairs passing cuts required

      typedef std::vector<T1> T1Collection;
      typedef edm::Ref<T1Collection> T1Ref;
      std::vector<T1Ref> coll1_;
      typedef std::vector<T2> T2Collection;
      typedef edm::Ref<T2Collection> T2Ref;
      std::vector<T2Ref> coll2_;


};

#endif //HLTDoubletDZ_h

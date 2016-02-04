#ifndef HLTRFilter_h
#define HLTRFilter_h

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include<vector>
#include "TLorentzVector.h"

namespace edm {
   class ConfigurationDescriptions;
}

//
// class declaration
//

class HLTRFilter : public HLTFilter {

   public:

      explicit HLTRFilter(const edm::ParameterSet&);
      ~HLTRFilter();
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      edm::InputTag inputTag_; // input tag identifying product
      edm::InputTag inputMetTag_; // input tag identifying MET product
      bool saveTag_;           // whether to save this tag
      double min_R_;           // minimum R vaule
      double min_MR_;          // minimum MR vaule
      bool DoRPrime_;          // Do the R' instead of R
      bool accept_NJ_;         // accept or reject events with high NJ

};

#endif //HLTRFilter_h

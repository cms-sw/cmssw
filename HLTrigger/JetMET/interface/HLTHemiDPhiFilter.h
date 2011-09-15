#ifndef HLTHemiDPhiFilter_h
#define HLTHemiDPhiFilter_h

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include<vector>
#include "TLorentzVector.h"

namespace edm {
   class ConfigurationDescriptions;
}

//
// class declaration
//

class HLTHemiDPhiFilter : public HLTFilter {

   public:

      explicit HLTHemiDPhiFilter(const edm::ParameterSet&);
      ~HLTHemiDPhiFilter();
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      

   private:
      double deltaPhi(double, double); //helper function
  
      edm::InputTag inputTag_; // input tag identifying product
      bool saveTags_;           // whether to save this tag
      double min_dphi_;          // minimum dphi value
      bool accept_NJ_;         // accept or reject events with high NJ

};

#endif //HLTHemiDPhiFilter_h

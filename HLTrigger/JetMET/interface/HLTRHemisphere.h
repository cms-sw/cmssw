#ifndef HLTRHemisphere_h
#define HLTRHemisphere_h

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include<vector>
#include "TLorentzVector.h"

namespace edm {
   class ConfigurationDescriptions;
}

//
// class declaration
//

class HLTRHemisphere : public HLTFilter {

   public:

      explicit HLTRHemisphere(const edm::ParameterSet&);
      ~HLTRHemisphere();
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct);

   private:
      edm::InputTag inputTag_; // input tag identifying product
      double min_Jet_Pt_;      // minimum jet pT threshold for collection
      double max_Eta_;         // maximum eta
      int max_NJ_;             // don't calculate R if event has more than NJ jets
      bool accNJJets_;         // accept or reject events with high NJ
};

#endif //HLTRHemisphere_h

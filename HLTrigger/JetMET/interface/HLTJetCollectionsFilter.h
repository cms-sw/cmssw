#ifndef HLTJetCollectionsFilter_h
#define HLTJetCollectionsFilter_h

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

namespace edm {
   class ConfigurationDescriptions;
}

//
// class declaration
//

class HLTJetCollectionsFilter : public HLTFilter {

   public:
      explicit HLTJetCollectionsFilter(const edm::ParameterSet&);
      ~HLTJetCollectionsFilter();
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct);

   private:
      edm::InputTag inputTag_; // input tag identifying jet collections
      edm::InputTag originalTag_; // input tag original jet collection
      double minJetPt_; // jet pt threshold in GeV
      double maxAbsJetEta_; // jet |eta| range
      unsigned int minNJets_; // number of required jets passing cuts after cleaning
};

#endif //HLTJetCollectionsFilter_h

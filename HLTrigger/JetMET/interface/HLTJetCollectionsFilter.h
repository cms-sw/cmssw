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
      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      edm::InputTag inputTag_; // input tag identifying jet collections
      bool saveTags_;              // whether to save this tag

      double minJetPt_; // jet pt threshold in GeV
      double maxAbsJetEta_; // jet |eta| range
      unsigned int minNJets_; // number of required jets passing cuts after cleaning


};

#endif //HLTJetCollectionsFilter_h

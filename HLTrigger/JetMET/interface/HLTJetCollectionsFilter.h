#ifndef HLTJetCollectionsFilter_h
#define HLTJetCollectionsFilter_h

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace edm {
   class ConfigurationDescriptions;
}

//
// class declaration
//

template <typename jetType>
class HLTJetCollectionsFilter : public HLTFilter {

   public:
      explicit HLTJetCollectionsFilter(const edm::ParameterSet&);
      ~HLTJetCollectionsFilter();
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
      bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct);
   private:
      edm::InputTag inputTag_; // input tag identifying jet collections
      edm::InputTag originalTag_; // input tag original jet collection
      double minJetPt_; // jet pt threshold in GeV
      double maxAbsJetEta_; // jet |eta| range
      unsigned int minNJets_; // number of required jets passing cuts after cleaning
      int triggerType_;
};

#endif //HLTJetCollectionsFilter_h

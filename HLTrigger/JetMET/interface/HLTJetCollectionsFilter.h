#ifndef HLTJetCollectionsFilter_h
#define HLTJetCollectionsFilter_h

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
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
      bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;
   private:
      edm::InputTag inputTag_; // input tag identifying jet collections
      edm::InputTag originalTag_; // input tag original jet collection
      double minJetPt_; // jet pt threshold in GeV
      double maxAbsJetEta_; // jet |eta| range
      unsigned int minNJets_; // number of required jets passing cuts after cleaning
      int triggerType_;
      edm::EDGetTokenT<std::vector<edm::RefVector<std::vector<jetType>,jetType,edm::refhelper::FindUsingAdvance<std::vector<jetType>,jetType> > >> m_theJetToken;
};

#endif //HLTJetCollectionsFilter_h

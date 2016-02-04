#ifndef HLTJetCollectionsVBFFilter_h
#define HLTJetCollectionsVBFFilter_h

/** \class HLTJetCollectionsVBFFilter
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

namespace edm {
   class ConfigurationDescriptions;
}

//
// class decleration
//

class HLTJetCollectionsVBFFilter : public HLTFilter {

   public:
      explicit HLTJetCollectionsVBFFilter(const edm::ParameterSet&);
      ~HLTJetCollectionsVBFFilter();
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      edm::InputTag inputTag_; // input tag identifying jets
      bool saveTags_;           // whether to save this tag
      double softJetPt_;
      double hardJetPt_;
      double minDeltaEta_;
      double thirdJetPt_;
      double maxAbsJetEta_;
      double maxAbsThirdJetEta_;
      unsigned int minNJets_; // number of required jets passing cuts after cleaning

};

#endif //HLTJetCollectionsVBFFilter_h

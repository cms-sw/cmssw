#ifndef HLTEgammaTriggerFilterObjectWrapper_h
#define HLTEgammaTriggerFilterObjectWrapper_h

/** \class HLTEgammaTriggerFilterObjectWrapper
 *
 *  \author Alessio Ghezzi
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"

namespace edm {
  class ConfigurationDescriptions;
}

//
// class decleration
//

class HLTEgammaTriggerFilterObjectWrapper : public HLTFilter {

   public:
      explicit HLTEgammaTriggerFilterObjectWrapper(const edm::ParameterSet&);
      ~HLTEgammaTriggerFilterObjectWrapper() override;
      bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

   private:
      edm::EDGetTokenT<reco::RecoEcalCandidateCollection> candIsolatedToken_;
      edm::EDGetTokenT<reco::RecoEcalCandidateCollection> candNonIsolatedToken_;
      edm::InputTag candIsolatedTag_; // input tag identifying product contains egammas
      edm::InputTag candNonIsolatedTag_; // input tag identifying product contains egammas
      bool doIsolated_;
};

#endif //HLTEgammaTriggerFilterObjectWrapper_h

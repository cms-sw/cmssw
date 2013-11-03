#ifndef HLTrigger_Egamma_HLTEgammaAllCombMassFilter_h
#define HLTrigger_Egamma_HLTEgammaAllCombMassFilter_h

//Class: HLTEgammaAllCombMassFilter
//purpose: the last filter of multi-e/g triggers which have asymetric cuts on the e/g objects
//         this checks that the required number of pair candidate pass a minimum mass cut

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "DataFormats/Math/interface/LorentzVector.h"

namespace edm {
  class ConfigurationDescriptions;
}

class HLTEgammaAllCombMassFilter : public HLTFilter {

 public:
  explicit HLTEgammaAllCombMassFilter(const edm::ParameterSet&);
  ~HLTEgammaAllCombMassFilter();
  virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;
  static void getP4OfLegCands(const edm::Event& iEvent, const edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs>& filterToken, std::vector<math::XYZTLorentzVector>& p4s);
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

 private:
  edm::InputTag firstLegLastFilterTag_;
  edm::InputTag secondLegLastFilterTag_;
  edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> firstLegLastFilterToken_;
  edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> secondLegLastFilterToken_;
  double minMass_;
};

#endif



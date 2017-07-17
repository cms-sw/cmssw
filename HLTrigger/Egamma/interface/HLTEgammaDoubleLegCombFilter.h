#ifndef HLTrigger_Egamma_HLTEgammaDoubleLegCombFilter_h
#define HLTrigger_Egamma_HLTEgammaDoubleLegCombFilter_h

//Class: HLTEgammaDoubleLegCombFilter
//purpose: the last filter of multi-e/g triggers which have asymetric cuts on the e/g objects
//         this checks that the required number of objects pass each leg

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "DataFormats/Math/interface/Point3D.h"


namespace edm {
  class ConfigurationDescriptions;
}

class HLTEgammaDoubleLegCombFilter : public HLTFilter {

 public:
  explicit HLTEgammaDoubleLegCombFilter(const edm::ParameterSet&);
  ~HLTEgammaDoubleLegCombFilter();
  virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

  void matchCands(const std::vector<math::XYZPoint>& firstLegP3s,const std::vector<math::XYZPoint>& secondLegP3s,std::vector<std::pair<int,int> >&matchedCands) const;
  static void getP3OfLegCands(const edm::Event& iEvent, const edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs>& filterToken, std::vector<math::XYZPoint>& p3s);

 private:
  edm::InputTag firstLegLastFilterTag_;
  edm::InputTag secondLegLastFilterTag_;
  edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> firstLegLastFilterToken_;
  edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> secondLegLastFilterToken_;
  int nrRequiredFirstLeg_;
  int nrRequiredSecondLeg_;
  int nrRequiredUniqueLeg_;
  double maxMatchDR_;
};

#endif



#ifndef HLTrigger_Egamma_HLTEgammaDoubleLegCombFilter_h
#define HLTrigger_Egamma_HLTEgammaDoubleLegCombFilter_h

//Class: HLTEgammaDoubleLegCombFilter
//purpose: the last filter of multi-e/g triggers which have asymetric cuts on the e/g objects
//         this checks that the required number of objects pass each leg

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include "DataFormats/Math/interface/Point3D.h"


class HLTEgammaDoubleLegCombFilter : public HLTFilter {

 public:
  explicit HLTEgammaDoubleLegCombFilter(const edm::ParameterSet&);
  ~HLTEgammaDoubleLegCombFilter();
  virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct);
  void matchCands(const std::vector<math::XYZPoint>& firstLegP3s,const std::vector<math::XYZPoint>& secondLegP3s,std::vector<std::pair<int,int> >&matchedCands);
  static void getP3OfLegCands(const edm::Event& iEvent,const edm::InputTag& filterTag,std::vector<math::XYZPoint>& p3s);
  
 private:
  edm::InputTag firstLegLastFilterTag_;
  edm::InputTag secondLegLastFilterTag_;
  int nrRequiredFirstLeg_;
  int nrRequiredSecondLeg_;
  int nrRequiredUniqueLeg_;
  double maxMatchDR_;
};

#endif 



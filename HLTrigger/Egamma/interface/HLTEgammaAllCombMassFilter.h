#ifndef HLTrigger_Egamma_HLTEgammaAllCombMassFilter_h
#define HLTrigger_Egamma_HLTEgammaAllCombMassFilter_h

//Class: HLTEgammaAllCombMassFilter
//purpose: the last filter of multi-e/g triggers which have asymetric cuts on the e/g objects
//         this checks that the required number of pair candidate pass a minimum mass cut

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include "DataFormats/Math/interface/LorentzVector.h"


class HLTEgammaAllCombMassFilter : public HLTFilter {

 public:
  explicit HLTEgammaAllCombMassFilter(const edm::ParameterSet&);
  ~HLTEgammaAllCombMassFilter();
  virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct);
  static void getP4OfLegCands(const edm::Event& iEvent,edm::InputTag filterTag,std::vector<math::XYZTLorentzVector>& p4s);
  
 private:
  edm::InputTag firstLegLastFilterTag_;
  edm::InputTag secondLegLastFilterTag_;
  double minMass_;
};

#endif 



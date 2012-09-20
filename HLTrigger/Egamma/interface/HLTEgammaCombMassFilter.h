#ifndef HLTrigger_Egamma_HLTEgammaCombMassFilter_h
#define HLTrigger_Egamma_HLTEgammaCombMassFilter_h

//Class: HLTEgammaCombMassFilter
//purpose: the last filter of multi-e/g triggers which have asymetric cuts on the e/g objects
//         this checks that the required number of pair candidate pass a minimum mass cut

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include "DataFormats/Math/interface/LorentzVector.h"


class HLTEgammaCombMassFilter : public HLTFilter {

 public:
  explicit HLTEgammaCombMassFilter(const edm::ParameterSet&);
  ~HLTEgammaCombMassFilter();
  virtual bool filter(edm::Event&, const edm::EventSetup&);
  static void getP4OfLegCands(const edm::Event& iEvent,edm::InputTag filterTag,std::vector<math::XYZTLorentzVector>& p4s);
  
 private:
  edm::InputTag firstLegLastFilterTag_;
  edm::InputTag secondLegLastFilterTag_;
  double minMass_;
};

#endif 



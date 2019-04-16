#ifndef PhysicsTools_PatAlgos_AuxiliaryProd_h
#define PhysicsTools_PatAlgos_AuxiliaryProd_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "PhysicsTools/PatUtils/interface/StringParserTools.h"
#include "PhysicsTools/PatUtils/interface/PATDiObjectProxy.h"

namespace pat{

  //
  // Hcal depth energy fracrtion struct
  //
  struct HcalDepthEnergyFractionProd
  {
  private:
    //do not store
    std::vector<float> hcalDepthEnergyFraction_;
  public:
    explicit HcalDepthEnergyFractionProd(std::vector<float> v):hcalDepthEnergyFraction_(v),hcalDepthEnergyFractionI_(std::vector<uint8_t>()) { initUint8(); }
    HcalDepthEnergyFractionProd():hcalDepthEnergyFraction_(std::vector<float>()),hcalDepthEnergyFractionI_(std::vector<uint8_t>()) { }
    std::vector<uint8_t> hcalDepthEnergyFractionI_;

    // produce vector of uint8 vector from vector of float
    void initUint8() {
      for (auto it = begin (hcalDepthEnergyFraction_); it != end (hcalDepthEnergyFraction_); ++it) 
	hcalDepthEnergyFractionI_.push_back((uint8_t)((*it)*200.));
    }

    // reset vector
    void reset(std::vector<float> v) {
      hcalDepthEnergyFraction_.clear();
      hcalDepthEnergyFractionI_.clear();
      for (auto it = begin (v); it != end (v); ++it){ 
	hcalDepthEnergyFractionI_.push_back((uint8_t)((*it)*200.));
	hcalDepthEnergyFraction_.push_back(*it);
      }
    }

    // provide a full vector for each depth
    std::vector<float> hcalDepthEnergyFraction() {
      hcalDepthEnergyFraction_.clear();
      for (auto it = begin (hcalDepthEnergyFractionI_); it != end (hcalDepthEnergyFractionI_); ++it)
	hcalDepthEnergyFraction_.push_back(float(*it)/200.);
      return hcalDepthEnergyFraction_;
    }

    // provide the info for each depth
    float hcalDepthEnergyFraction(unsigned int i) {
      if (i<hcalDepthEnergyFractionI_.size())
	return float(hcalDepthEnergyFractionI_[i])/200.;
      else return -1.;
    }
    
  };

}
 
#endif


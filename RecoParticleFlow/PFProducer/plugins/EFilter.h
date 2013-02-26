#ifndef RecoParticleFlow_PFProducer_EFilter_h_
#define RecoParticleFlow_PFProducer_EFilter_h_

#include "FWCore/Framework/interface/EDFilter.h"
// #include "FWCore/ParameterSet/interface/ParameterSet.h"


// -*- C++ -*-
//
// Package:    EFilter
// Class:      EFilter
// 
/**\class EFilter 

 Description: filters single particle events according to the energy of 
 the mother particle

*/



//
// class declaration
//

class FSimEvent;

class EFilter : public edm::EDFilter {
 public:
  explicit EFilter(const edm::ParameterSet&);
  ~EFilter();

 private:
  virtual bool filter(edm::Event&, const edm::EventSetup&) override;
  
  // ----------member data ---------------------------
/*   edm::ParameterSet  vertexGenerator_;  */
/*   edm::ParameterSet  particleFilter_; */
/*   FSimEvent* mySimEvent; */
  // std::string hepMCModuleLabel_;

  edm::InputTag  inputTagParticles_;

  double minE_;
  double maxE_;
  double minEt_;
  double maxEt_;
};

#endif

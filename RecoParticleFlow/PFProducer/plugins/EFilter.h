#ifndef RecoParticleFlow_PFProducer_EFilter_h_
#define RecoParticleFlow_PFProducer_EFilter_h_

#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "DataFormats/ParticleFlowReco/interface/PFSimParticle.h"

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

class EFilter : public edm::stream::EDFilter<> {
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

  edm::EDGetTokenT<std::vector<reco::PFSimParticle> > inputTagParticles_;

  double minE_;
  double maxE_;
  double minEt_;
  double maxEt_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(EFilter);

#endif

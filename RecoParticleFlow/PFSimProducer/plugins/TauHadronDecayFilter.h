#ifndef RecoParticleFlow_PFProducer_TauHadronDecayFilter_h_
#define RecoParticleFlow_PFProducer_TauHadronDecayFilter_h_

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FastSimulation/Particle/interface/ParticleTable.h"

// -*- C++ -*-
//
// Package:    TauHadronDecayFilter
// Class:      TauHadronDecayFilter
// 
/**\class TauHadronDecayFilter 

 Description: filters single tau events with a tau decaying hadronically
*/
//
// Original Author:  Colin BERNET
//         Created:  Mon Nov 13 11:06:39 CET 2006
//
//


//
// class declaration
//

class FSimEvent;

class TauHadronDecayFilter : public edm::EDFilter {
 public:
  explicit TauHadronDecayFilter(const edm::ParameterSet&);
  ~TauHadronDecayFilter();

 private:
  virtual void beginRun(const edm::Run&, const edm::EventSetup&) override;
  virtual bool filter(edm::Event&, const edm::EventSetup&) override;
  
  // ----------member data ---------------------------
  edm::ParameterSet  particleFilter_;
  FSimEvent* mySimEvent;
  std::unique_ptr<ParticleTable::Sentry> pTableSentry_;
};

#endif

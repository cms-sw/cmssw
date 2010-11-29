#ifndef RecoParticleFlow_PFProducer_TauHadronDecayFilter_h_
#define RecoParticleFlow_PFProducer_TauHadronDecayFilter_h_

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


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
// $Id: TauHadronDecayFilter.h,v 1.1 2009/05/06 14:20:25 pjanot Exp $
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
  virtual bool beginRun(edm::Run&, const edm::EventSetup&) ;
  virtual bool filter(edm::Event&, const edm::EventSetup&);
  virtual void endRun() ;
  
  // ----------member data ---------------------------
  edm::ParameterSet  vertexGenerator_; 
  edm::ParameterSet  particleFilter_;
  FSimEvent* mySimEvent;
};

#endif

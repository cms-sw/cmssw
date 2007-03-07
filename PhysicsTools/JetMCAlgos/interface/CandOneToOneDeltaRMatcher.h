#ifndef CANDONETOONEDELTARMATCHER_H
#define CANDONETOONEDELTARMATCHER_H

/* \class CandOneToOneDeltaRMatcher
 *
 * Producer for simple match map
 * to match two collections of candidate
 * with one-to-One matching 
 * minimizing Sum(DeltaR)
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include<vector>
#include<iostream>

class CandOneToOneDeltaRMatcher : public edm::EDProducer {
 public:
  CandOneToOneDeltaRMatcher( const edm::ParameterSet & );
  ~CandOneToOneDeltaRMatcher();
 private:
  void produce( edm::Event&, const edm::EventSetup& );
  double lenght( std::vector<int> );
  
  edm::InputTag source_;
  edm::InputTag matched_;
  std::vector < std::vector<float> > AllDist;
  bool printdebug_;

};

#endif


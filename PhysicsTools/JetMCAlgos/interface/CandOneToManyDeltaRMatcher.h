#ifndef CANDONETOMANYDELTARMATCHER_H
#define CANDONETOMANYDELTARMATCHER_H

/* \class CandOneToManyDeltaRMatcher
 *
 * Producer for simple match map:
 * class to match two collections of candidate
 * with one-to-Many matching 
 * All elements of class "matched" are matched to each element
 * of class "source" orderd in DeltaR
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include<vector>
#include<iostream>

class CandOneToManyDeltaRMatcher : public edm::EDProducer {
 public:
  CandOneToManyDeltaRMatcher( const edm::ParameterSet & );
  ~CandOneToManyDeltaRMatcher();
 private:
  void produce( edm::Event&, const edm::EventSetup& );
  
  edm::InputTag source_;
  edm::InputTag matched_;
  bool printdebug_;
};

#endif


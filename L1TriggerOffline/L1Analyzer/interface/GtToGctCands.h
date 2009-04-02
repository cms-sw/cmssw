#ifndef L1Analyzer_GtToGctCands_h
#define L1Analyzer_GtToGctCands_h

// -*- C++ -*-
//
// Package:    GtToGctCands
// Class:      GtToGctCands
// 
/**\class GtToGctCands GtToGctCands.cc L1TriggerOffline/L1Analyzer/interface/GtToGctCands.h

Description: Convert GT candidates (electrons and jets) to GCT format

*/
//
// Original Author:  Alex Tapper
//         Created:  Mon Mar 30 17:31:03 CEST 2009
// $Id: GtToGctCands.h,v 1.2 2009/04/02 12:01:48 tapper Exp $
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"

class GtToGctCands : public edm::EDProducer {

 public:
  explicit GtToGctCands(const edm::ParameterSet&);
  ~GtToGctCands();

 private:
  virtual void produce(edm::Event&, const edm::EventSetup&);

  edm::InputTag m_GTInputTag;
      
};

#endif

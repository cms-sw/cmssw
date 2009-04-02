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
// $Id: GtToGctCands.h,v 1.1 2009/04/01 12:32:05 tapper Exp $
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/OrphanHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Math/interface/LorentzVector.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"

#include "CondFormats/L1TObjects/interface/L1CaloGeometry.h"
#include "CondFormats/DataRecord/interface/L1CaloGeometryRecord.h"

// forward declarations
class L1CaloGeometry ;

class GtToGctCands : public edm::EDProducer {

 public:
  explicit GtToGctCands(const edm::ParameterSet&);
  ~GtToGctCands();

 private:
  virtual void produce(edm::Event&, const edm::EventSetup&);

  math::PtEtaPhiMLorentzVector gctLorentzVector( const double& et,
                                                 const L1GctCand& cand,
                                                 const L1CaloGeometry* geom,
                                                 bool central ) ;

  edm::InputTag m_GTInputTag;
      
};

#endif

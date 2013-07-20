#ifndef RecoJets_JetProducers_plugins_HiL1Subtractor_h
#define RecoJets_JetProducers_plugins_HiL1Subtractor_h

// -*- C++ -*-
//
// Package:    HiL1Subtractor
// Class:      HiL1Subtractor
//
/**\class HiL1Subtractor HiL1Subtractor.cc RecoHI/HiJetAlgos/plugins/HiL1Subtractor.cc

 Description:

  Implementation:

*/
//
// Original Author:  "Matthew Nguyen"
//         Created:  Sun Nov 7 12:18:18 CDT 2010
// $Id: HiL1Subtractor.h,v 1.1 2011/01/10 19:56:55 mnguyen Exp $
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
#include "FWCore/Utilities/interface/InputTag.h"


//
// class declaration
//

class HiL1Subtractor : public edm::EDProducer {

 protected:
  //
  // typedefs & structs
  //

 public:

  explicit HiL1Subtractor(const edm::ParameterSet&);
  ~HiL1Subtractor();


 private:
  virtual void beginJob() ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;


  // ----------member data ---------------------------
  edm::InputTag                 src_;         // input jet source

 protected:
  std::string                   jetType_;     // Type of jet
  std::string                   rhoTag_;     // Algorithm for rho estimation
};


#endif

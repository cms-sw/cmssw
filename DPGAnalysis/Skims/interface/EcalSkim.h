// -*- C++ -*-
//
// Package:   EcalSkim
// Class:     EcalSkim
//
//class EcalSkim EcalSkim.cc
//
// Original Author:  Serena OGGERO
//         Created:  We May 14 10:10:52 CEST 2008
//        Modified:  Toyoko ORIMOTO
//

#ifndef EcalSkim_H
#define EcalSkim_H

// system include files
#include <memory>
#include <vector>
#include <map>
#include <set>

// user include files
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "TFile.h"
#include <string>

//
// class declaration
//

class TFile;

class EcalSkim : public edm::EDFilter {
public:
  explicit EcalSkim( const edm::ParameterSet & );
  ~EcalSkim();
  
private:
  virtual bool filter ( edm::Event &, const edm::EventSetup&) override;
  
  edm::InputTag BarrelClusterCollection;
  edm::InputTag EndcapClusterCollection;
  double EnergyCutEB;
  double EnergyCutEE;
  
};

#endif

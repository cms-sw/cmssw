// -*- C++ -*-
//
// Package:   EcalHighEnCosmicFilter
// Class:     EcalHighEnCosmicFilter
//
//class EcalHighEnCosmicFilter EcalHighEnCosmicFilter.cc
//
// Original Author:  Serena OGGERO
//         Created:  We May 14 10:10:52 CEST 2008
//

#ifndef EcalHighEnCosmicFilter_H
#define EcalHighEnCosmicFilter_H


// system include files
#include <memory>
#include <vector>
#include <map>
#include <set>

// user include files
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "TFile.h"
#include <string>
#include <TFile.h>

//
// class declaration
//

class TFile;

class EcalHighEnCosmicFilter : public edm::EDFilter {
   public:
      explicit EcalHighEnCosmicFilter(const edm::ParameterSet &);
      ~EcalHighEnCosmicFilter();

   private:
      virtual bool filter ( edm::Event &, const edm::EventSetup &);

      int nHighEnClus;
      int nEvent;
      int nGoodEvent;
     
      edm::InputTag BarrelClusterCollection;
      edm::InputTag EndcapClusterCollection;
      double EnergyCut;

};

#endif

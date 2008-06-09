#ifndef CLUSTERCOUNT_H
#define CLUSTERCOUNT_H
// -*- C++ -*-
//
// Package:    ClusterCount
// Class:      ClusterCount
// 
/**\class ClusterCount ClusterCount.cc mytests/ClusterCount/src/ClusterCount.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Christophe DELAERE
//         Created:  Tue May 27 11:11:05 CEST 2008
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiStripCluster/interface/SiStripClusterCollection.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include <DataFormats/Common/interface/DetSetVector.h>

//
// class decleration
//

class ClusterCount : public edm::EDAnalyzer {
   public:
      explicit ClusterCount(const edm::ParameterSet&);
      ~ClusterCount();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------
      edm::InputTag clusterLabel_;
};

#endif


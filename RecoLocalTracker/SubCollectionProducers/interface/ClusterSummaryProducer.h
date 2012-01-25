#ifndef CLUSTERSUMMARYPRODUCER
#define CLUSTERSUMMARYPRODUCER

// -*- C++ -*-
//
// Package:    ClusterSummaryProducer
// Class:      ClusterSummaryProducer
// 
/**\class ClusterSummaryProducer ClusterSummaryProducer.cc msegala/ClusterSummaryProducer/src/ClusterSummaryProducer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Michael Segala
//         Created:  Thu Jun 23 09:33:08 CDT 2011
// $Id: ClusterSummaryProducer.h,v 1.1 2012/01/24 18:00:23 msegala Exp $
//
//


// system include files
#include <memory>
#include <string>
#include <map>
#include <vector>
#include<iostream>
#include <string.h>
// user include files

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"


#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterInfo.h"
#include "DataFormats/SiStripDigi/interface/SiStripProcessedRawDigi.h"
#include "DataFormats/Common/interface/DetSetVector.h"


#include "DataFormats/TrackerCommon/interface/ClusterSummary.h"
#include "RecoLocalTracker/SubCollectionProducers/interface/ClusterVariables.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/DetId/interface/DetId.h" 
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"


//
// class declaration
//
class ClusterVariables;
class ClusterSummary;


class ClusterSummaryProducer : public edm::EDProducer {
   public:
      explicit ClusterSummaryProducer(const edm::ParameterSet&);
      ~ClusterSummaryProducer(){};

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      
      // ----------member data ---------------------------
      
      edm::InputTag stripClustersLabel;
      std::string modules;
      std::vector<std::string> v_moduleTypes;
      
      std::string variables;
      std::vector<std::string> v_variables;
      
      ClusterSummary cCluster;
      std::map< std::string, int > EnumMap;
      std::vector<ClusterSummary::ModuleSelection*> ModuleSelectionVect;

      bool verbose;
      bool firstpass;
      bool firstpass_mod;

      //Declare the variables to fill the summary info with
      std::vector<std::string> v_userContent; 
      
      

};

#endif

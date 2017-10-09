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
#include "FWCore/Framework/interface/stream/EDProducer.h"
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
#include "CommonTools/UtilAlgos/interface/DetIdSelector.h"


//
// class declaration
//
class ClusterVariables;
class ClusterSummary;


class ClusterSummaryProducer : public edm::stream::EDProducer<> {
   public:
      explicit ClusterSummaryProducer(const edm::ParameterSet&);
      ~ClusterSummaryProducer(){};

   private:
      virtual void beginStream(edm::StreamID) override;
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      

      typedef std::pair<DetIdSelector,ClusterSummary::CMSTracker> ModuleSelection;
      typedef std::vector<ModuleSelection> ModuleSelections;

      // ----------member data ---------------------------
      
      edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster> > pixelClusters_;
      edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster> > stripClusters_;

      ModuleSelections         selectors;
      std::vector<std::string> moduleNames;

      ClusterSummary cCluster;
      std::map< std::string, int > EnumMap;


      bool doStrips;
      bool doPixels;
      bool verbose;

};

#endif

// -*- C++ -*-
//
// Package:    OuterTrackerMonitorCluster
// Class:      OuterTrackerMonitorCluster
//
/**\class OuterTrackerMonitorCluster OuterTrackerMonitorCluster.cc DQM/OuterTrackerMonitorCluster/plugins/OuterTrackerMonitorCluster.cc
 
 Description: [one line class summary]
 
 Implementation:
 [Notes on implementation]
 */
//
// Original Author:  Isabelle Helena J De Bruyn
//         Created:  Mon, 10 Feb 2014 13:57:08 GMT
//

// system include files
#include <memory>
#include <vector>
#include <numeric>
#include <fstream>
#include <math.h>
#include "TNamed.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "DataFormats/SiStripCluster/interface/SiStripClusterCollection.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/SiStripCommon/interface/SiStripHistoId.h"
#include "DQM/OuterTrackerMonitorCluster/interface/OuterTrackerMonitorCluster.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/SiStripSubStructure.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDCSStatus.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"

#include "DPGAnalysis/SiStripTools/interface/APVCyclePhaseCollection.h"
#include "DPGAnalysis/SiStripTools/interface/EventWithHistory.h"

#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"

// For TPart_Eta_ICW_1 (TrackingParticles)
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTClusterAssociationMap.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTCluster.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"


#include "TMath.h"
#include <iostream>

//
// constructors and destructor
//
OuterTrackerMonitorCluster::OuterTrackerMonitorCluster(const edm::ParameterSet& iConfig)
: dqmStore_(edm::Service<DQMStore>().operator->()), conf_(iConfig)

{
  clusterProducerStrip_ = conf_.getParameter<edm::InputTag>("ClusterProducerStrip");
  topFolderName_ = conf_.getParameter<std::string>("TopFolderName");
  
}


OuterTrackerMonitorCluster::~OuterTrackerMonitorCluster()
{
	
	// do anything here that needs to be done at desctruction time
	// (e.g. close files, deallocate resources etc.)
	
}


//
// member functions
//

// ------------ method called for each event  ------------
void
OuterTrackerMonitorCluster::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::Handle< edmNew::DetSetVector<SiStripCluster> > cluster_detsetvektor;
  iEvent.getByLabel(clusterProducerStrip_, cluster_detsetvektor);
  int NStripClusters=0;
  if (!cluster_detsetvektor.isValid()) return;
  const edmNew::DetSetVector<SiStripCluster> * StrC= cluster_detsetvektor.product();
  NStripClusters= StrC->data().size();
  NumberOfStripClus->Fill(NStripClusters);
	
}


// ------------ method called once each job just before starting event loop  ------------
void
OuterTrackerMonitorCluster::beginRun(const edm::Run& run, const edm::EventSetup& es)
{
	
	SiStripFolderOrganizer folder_organizer;
	folder_organizer.setSiStripFolderName(topFolderName_);
	folder_organizer.setSiStripFolder();
	
	dqmStore_->setCurrentFolder(topFolderName_+"/MechanicalView/");
	
	edm::ParameterSet StripCluster =  conf_.getParameter<edm::ParameterSet>("TH1NClusStrip");
	std::string HistoName = "NumberOfClustersInStrip";
	NumberOfStripClus = dqmStore_->book1D(HistoName, HistoName,
																				StripCluster.getParameter<int32_t>("Nbinsx"),
																				StripCluster.getParameter<double>("xmin"),
																				StripCluster.getParameter<double>("xmax"));
	NumberOfStripClus->setAxisTitle("# of Clusters in Strip", 1);
	NumberOfStripClus->setAxisTitle("Number of Events", 2);
	
	
}//end of method


// ------------ method called once each job just after ending the event loop  ------------
void 
OuterTrackerMonitorCluster::endJob(void) 
{
	
}

DEFINE_FWK_MODULE(OuterTrackerMonitorCluster);

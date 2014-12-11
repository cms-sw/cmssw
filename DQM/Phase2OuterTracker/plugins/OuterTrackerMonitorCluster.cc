// -*- C++ -*-
//
// Package:    Phase2OuterTracker
// Class:      Phase2OuterTracker
//
/**\class Phase2OuterTracker OuterTrackerMonitorCluster.cc DQM/Phase2OuterTracker/plugins/OuterTrackerMonitorCluster.cc
 
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
#include <iostream>
#include <fstream>
#include <math.h>
#include "TMath.h"
#include "TNamed.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/Phase2OuterTracker/interface/OuterTrackerMonitorCluster.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTCluster.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerGeometry.h"
#include "Geometry/Records/interface/StackedTrackerGeometryRecord.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"

//
// constructors and destructor
//
OuterTrackerMonitorCluster::OuterTrackerMonitorCluster(const edm::ParameterSet& iConfig)
: dqmStore_(edm::Service<DQMStore>().operator->()), conf_(iConfig)
{
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
void OuterTrackerMonitorCluster::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
	/// Track Trigger Clusters
	edm::Handle< edmNew::DetSetVector< TTCluster< Ref_PixelDigi_ > > > PixelDigiTTClusterHandle;
  iEvent.getByLabel( "TTClustersFromPixelDigis", "ClusterInclusive", PixelDigiTTClusterHandle );
  
  /// Geometry
  edm::ESHandle< StackedTrackerGeometry > StackedGeometryHandle;
  const StackedTrackerGeometry* theStackedGeometry;
  iSetup.get< StackedTrackerGeometryRecord >().get(StackedGeometryHandle);
  theStackedGeometry = StackedGeometryHandle.product();
  
  /// Loop over the input Clusters
	typename edmNew::DetSetVector< TTCluster< Ref_PixelDigi_ > >::const_iterator inputIter;
	typename edmNew::DetSet< TTCluster< Ref_PixelDigi_ > >::const_iterator contentIter;
	for ( inputIter = PixelDigiTTClusterHandle->begin();
			 inputIter != PixelDigiTTClusterHandle->end();
			 ++inputIter )
	{
		for(contentIter = inputIter->begin(); contentIter != inputIter->end(); ++contentIter)
		{
			edm::Ref< edmNew::DetSetVector< TTCluster< Ref_PixelDigi_ > >, TTCluster< Ref_PixelDigi_ > > tempCluRef = edmNew::makeRefTo( PixelDigiTTClusterHandle, contentIter );
			StackedTrackerDetId detIdClu( tempCluRef->getDetId() );
			unsigned int memberClu = tempCluRef->getStackMember();
			unsigned int widClu = tempCluRef->findWidth();
      
      GlobalPoint posClu  = theStackedGeometry->findAverageGlobalPosition( &(*tempCluRef) );
  
      double eta = posClu.eta();
      
			Cluster_W->Fill(widClu, memberClu);
      Cluster_Eta->Fill(eta);

			if ( detIdClu.isBarrel() )
			{
				if (memberClu == 0) Cluster_IMem_Barrel->Fill(detIdClu.iLayer());
				else Cluster_OMem_Barrel->Fill(detIdClu.iLayer());
			}	// end if isBarrel()
			else if (detIdClu.isEndcap())
			{
				if (memberClu == 0) Cluster_IMem_Endcap->Fill(detIdClu.iDisk());
				else Cluster_OMem_Endcap->Fill(detIdClu.iDisk());
			}	// end if isEndcap()
		}	// end loop contentIter
	}	// end loop inputIter
} // end of method

// ------------ method called once each job just before starting event loop  ------------
void
OuterTrackerMonitorCluster::beginRun(const edm::Run& run, const edm::EventSetup& es)
{
  SiStripFolderOrganizer folder_organizer;
	folder_organizer.setSiStripFolderName(topFolderName_);
	folder_organizer.setSiStripFolder();	
	
	dqmStore_->setCurrentFolder(topFolderName_+"/Clusters/");
	
	// NClusters
	edm::ParameterSet psTTClusterStacks =  conf_.getParameter<edm::ParameterSet>("TH1TTCluster_Stack");
	std::string HistoName = "NClusters_IMem_Barrel";
	Cluster_IMem_Barrel = dqmStore_->book1D(HistoName, HistoName,
	psTTClusterStacks.getParameter<int32_t>("Nbinsx"),
	psTTClusterStacks.getParameter<double>("xmin"),
	psTTClusterStacks.getParameter<double>("xmax"));
	Cluster_IMem_Barrel->setAxisTitle("Layer", 1);
	Cluster_IMem_Barrel->setAxisTitle("# TTClusters", 2);
	
	HistoName = "NClusters_IMem_Endcap";
	Cluster_IMem_Endcap = dqmStore_->book1D(HistoName, HistoName,
	psTTClusterStacks.getParameter<int32_t>("Nbinsx"),
	psTTClusterStacks.getParameter<double>("xmin"),
	psTTClusterStacks.getParameter<double>("xmax"));
	Cluster_IMem_Endcap->setAxisTitle("Disc", 1);
	Cluster_IMem_Endcap->setAxisTitle("# TTClusters", 2);
	
	HistoName = "NClusters_OMem_Barrel";
	Cluster_OMem_Barrel = dqmStore_->book1D(HistoName, HistoName,
	psTTClusterStacks.getParameter<int32_t>("Nbinsx"),
	psTTClusterStacks.getParameter<double>("xmin"),
	psTTClusterStacks.getParameter<double>("xmax"));
	Cluster_OMem_Barrel->setAxisTitle("Layer", 1);
	Cluster_OMem_Barrel->setAxisTitle("# TTClusters", 2);
	
	HistoName = "NClusters_OMem_Endcap";
	Cluster_OMem_Endcap = dqmStore_->book1D(HistoName, HistoName,
	psTTClusterStacks.getParameter<int32_t>("Nbinsx"),
	psTTClusterStacks.getParameter<double>("xmin"),
	psTTClusterStacks.getParameter<double>("xmax"));
	Cluster_OMem_Endcap->setAxisTitle("Disc", 1);
	Cluster_OMem_Endcap->setAxisTitle("# TTClusters", 2);
        
  //Cluster Width
  edm::ParameterSet psTTClusterWidth =  conf_.getParameter<edm::ParameterSet>("TH2TTCluster_Width");
  HistoName = "Cluster_W";
  Cluster_W = dqmStore_->book2D(HistoName, HistoName,
  psTTClusterWidth.getParameter<int32_t>("Nbinsx"),
  psTTClusterWidth.getParameter<double>("xmin"),
  psTTClusterWidth.getParameter<double>("xmax"),
  psTTClusterWidth.getParameter<int32_t>("Nbinsy"),
  psTTClusterWidth.getParameter<double>("ymin"),
  psTTClusterWidth.getParameter<double>("ymax"));
  Cluster_W->setAxisTitle("Cluster Width", 1);
  Cluster_W->setAxisTitle("Stack Member", 2);
  
  //Cluster eta distribution
  edm::ParameterSet psTTClusterEta = conf_.getParameter<edm::ParameterSet>("TH1TTCluster_Eta");
  HistoName = "Cluster_Eta";
  Cluster_Eta = dqmStore_->book1D(HistoName, HistoName, 
  psTTClusterEta.getParameter<int32_t>("Nbinsx"),
  psTTClusterEta.getParameter<double>("xmin"),
  psTTClusterEta.getParameter<double>("xmax"));
  Cluster_Eta->setAxisTitle("#eta", 1);
  Cluster_Eta->setAxisTitle("# TTClusters", 2);
                                  
}//end of method

// ------------ method called once each job just after ending the event loop  ------------
void 
OuterTrackerMonitorCluster::endJob(void) 
{
	
}

DEFINE_FWK_MODULE(OuterTrackerMonitorCluster);

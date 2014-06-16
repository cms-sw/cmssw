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
	
	
	/// TrackingParticles
	edm::Handle< std::vector< TrackingParticle > > TrackingParticleHandle;
	iEvent.getByLabel( "mix", "MergedTrackTruth", TrackingParticleHandle );
	/// Track Trigger
	edm::Handle< edmNew::DetSetVector< TTCluster< Ref_PixelDigi_ > > > PixelDigiTTClusterHandle;	// same for stubs
	iEvent.getByLabel( "TTClustersFromPixelDigis", "ClusterInclusive", PixelDigiTTClusterHandle );
	/// Track Trigger MC Truth
	edm::Handle< TTClusterAssociationMap< Ref_PixelDigi_ > > MCTruthTTClusterHandle;
	iEvent.getByLabel( "TTClusterAssociatorFromPixelDigis", "ClusterInclusive", MCTruthTTClusterHandle );
	
	/// Eta coverage
	/// Go on only if there are TrackingParticles
  if( TrackingParticleHandle->size() > 0)
  {
  	/// Loop over the TrackingParticles
		unsigned int tpCnt = 0;
		std::vector< TrackingParticle >::const_iterator iterTP;
		for(iterTP = TrackingParticleHandle->begin(); iterTP !=	TrackingParticleHandle->end(); ++iterTP)
		{
			/// Make the pointer
  		edm::Ptr<TrackingParticle> tempTPPtr( TrackingParticleHandle, tpCnt++ );
			
			/// Search the cluster MC map
			std::vector< edm::Ref< edmNew::DetSetVector< TTCluster< Ref_PixelDigi_ > >, TTCluster< Ref_PixelDigi_ > > > theseClusters = MCTruthTTClusterHandle->findTTClusterRefs( tempTPPtr );
			
		}	// end loop TrackingParticles
  } // end if there are TrackingParticles
	
	/// Loop over the input Clusters
	typename edmNew::DetSetVector< TTCluster< Ref_PixelDigi_ > >::const_iterator inputIter;
	typename edmNew::DetSet< TTCluster< Ref_PixelDigi_ > >::const_iterator contentIter;
	for ( inputIter = PixelDigiTTClusterHandle->begin();
			 inputIter != PixelDigiTTClusterHandle->end();
			 ++inputIter )
	{
		for ( contentIter = inputIter->begin();
				 contentIter != inputIter->end();
				 ++contentIter )
		{
			/// Make the reference to be put in the map
			edm::Ref< edmNew::DetSetVector< TTCluster< Ref_PixelDigi_ > >, TTCluster< Ref_PixelDigi_ > > tempCluRef = edmNew::makeRefTo( PixelDigiTTClusterHandle, contentIter );

			StackedTrackerDetId detIdClu( tempCluRef->getDetId() );		// find it!
			unsigned int memberClu = tempCluRef->getStackMember();
			bool genuineClu     = MCTruthTTClusterHandle->isGenuine( tempCluRef );
			bool combinClu      = MCTruthTTClusterHandle->isCombinatoric( tempCluRef );
			//bool unknownClu     = MCTruthTTClusterHandle->isUnknown( tempCluRef );
			//int partClu         = 999999999;
			if ( genuineClu )
			{
				edm::Ptr< TrackingParticle > thisTP = MCTruthTTClusterHandle->findTrackingParticlePtr( tempCluRef );
				//partClu = thisTP->pdgId();
			}

			if ( detIdClu.isBarrel() )
			{
				if ( memberClu == 0 )
				{
					Cluster_IMem_Barrel->Fill( detIdClu.iLayer() );
				}
				else
				{
					Cluster_OMem_Barrel->Fill( detIdClu.iLayer() );
				}

				if ( genuineClu )
				{
					Cluster_Gen_Barrel->Fill( detIdClu.iLayer() );
				}
				else if ( combinClu )
				{
					Cluster_Comb_Barrel->Fill( detIdClu.iLayer() );
				}
				else
				{
					Cluster_Unkn_Barrel->Fill( detIdClu.iLayer() );
				}

			}	// end if isBarrel()
			else if ( detIdClu.isEndcap() )
			{
				if ( memberClu == 0 )
				{
					Cluster_IMem_Endcap->Fill( detIdClu.iDisk() );
				}
				else
				{
					Cluster_OMem_Endcap->Fill( detIdClu.iDisk() );
				}

				if ( genuineClu )
				{
					Cluster_Gen_Endcap->Fill( detIdClu.iDisk() );
				}
				else if ( combinClu )
				{
					Cluster_Comb_Endcap->Fill( detIdClu.iDisk() );
				}
				else
				{
					Cluster_Unkn_Endcap->Fill( detIdClu.iDisk() );
				}

			}	// end if isEndcap()
		}	// end loop contentIter
	}	// end loop inputIter
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
	
	
	dqmStore_->setCurrentFolder(topFolderName_+"/Clusters/");
	
	// TTCluster stacks
	edm::ParameterSet psTTClusterStacks =  conf_.getParameter<edm::ParameterSet>("TH1TTCluster_Stack");
	HistoName = "Cluster_IMem_Barrel";
	Cluster_IMem_Barrel = dqmStore_->book1D(HistoName, HistoName,
																				psTTClusterStacks.getParameter<int32_t>("Nbinsx"),
																				psTTClusterStacks.getParameter<double>("xmin"),
																				psTTClusterStacks.getParameter<double>("xmax"));
	Cluster_IMem_Barrel->setAxisTitle("Inner TTCluster Stack", 1);
	Cluster_IMem_Barrel->setAxisTitle("Number of Clusters", 2);
	
	HistoName = "Cluster_IMem_Endcap";
	Cluster_IMem_Endcap = dqmStore_->book1D(HistoName, HistoName,
																				psTTClusterStacks.getParameter<int32_t>("Nbinsx"),
																				psTTClusterStacks.getParameter<double>("xmin"),
																				psTTClusterStacks.getParameter<double>("xmax"));
	Cluster_IMem_Endcap->setAxisTitle("Inner TTCluster Stack", 1);
	Cluster_IMem_Endcap->setAxisTitle("Number of Clusters", 2);
	
	HistoName = "Cluster_OMem_Barrel";
	Cluster_OMem_Barrel = dqmStore_->book1D(HistoName, HistoName,
																				psTTClusterStacks.getParameter<int32_t>("Nbinsx"),
																				psTTClusterStacks.getParameter<double>("xmin"),
																				psTTClusterStacks.getParameter<double>("xmax"));
	Cluster_OMem_Barrel->setAxisTitle("Outer TTCluster Stack", 1);
	Cluster_OMem_Barrel->setAxisTitle("Number of Clusters", 2);
	
	HistoName = "Cluster_IMem_Endcap";
	Cluster_OMem_Endcap = dqmStore_->book1D(HistoName, HistoName,
																				psTTClusterStacks.getParameter<int32_t>("Nbinsx"),
																				psTTClusterStacks.getParameter<double>("xmin"),
																				psTTClusterStacks.getParameter<double>("xmax"));
	Cluster_OMem_Endcap->setAxisTitle("Outer TTCluster Stack", 1);
	Cluster_OMem_Endcap->setAxisTitle("Number of Clusters", 2);
	
	HistoName = "Cluster_Gen_Barrel";
	Cluster_Gen_Barrel = dqmStore_->book1D(HistoName, HistoName,
																				psTTClusterStacks.getParameter<int32_t>("Nbinsx"),
																				psTTClusterStacks.getParameter<double>("xmin"),
																				psTTClusterStacks.getParameter<double>("xmax"));
	Cluster_Gen_Barrel->setAxisTitle("Genuine TTCluster Stack", 1);
	Cluster_Gen_Barrel->setAxisTitle("Number of Clusters", 2);
	
	HistoName = "Cluster_Unkn_Barrel";
	Cluster_Unkn_Barrel = dqmStore_->book1D(HistoName, HistoName,
																				psTTClusterStacks.getParameter<int32_t>("Nbinsx"),
																				psTTClusterStacks.getParameter<double>("xmin"),
																				psTTClusterStacks.getParameter<double>("xmax"));
	Cluster_Unkn_Barrel->setAxisTitle("Unknown TTCluster Stack", 1);
	Cluster_Unkn_Barrel->setAxisTitle("Number of Clusters", 2);
	
	HistoName = "Cluster_Comb_Barrel";
	Cluster_Comb_Barrel = dqmStore_->book1D(HistoName, HistoName,
																				psTTClusterStacks.getParameter<int32_t>("Nbinsx"),
																				psTTClusterStacks.getParameter<double>("xmin"),
																				psTTClusterStacks.getParameter<double>("xmax"));
	Cluster_Comb_Barrel->setAxisTitle("Combinatorial TTCluster Stack", 1);
	Cluster_Comb_Barrel->setAxisTitle("Number of Clusters", 2);
	
	HistoName = "Cluster_Gen_Endcap";
	Cluster_Gen_Endcap = dqmStore_->book1D(HistoName, HistoName,
																				psTTClusterStacks.getParameter<int32_t>("Nbinsx"),
																				psTTClusterStacks.getParameter<double>("xmin"),
																				psTTClusterStacks.getParameter<double>("xmax"));
	Cluster_Gen_Endcap->setAxisTitle("Genuine TTCluster Stack", 1);
	Cluster_Gen_Endcap->setAxisTitle("Number of Clusters", 2);
	
	HistoName = "Cluster_Unkn_Endcap";
	Cluster_Unkn_Endcap = dqmStore_->book1D(HistoName, HistoName,
																				psTTClusterStacks.getParameter<int32_t>("Nbinsx"),
																				psTTClusterStacks.getParameter<double>("xmin"),
																				psTTClusterStacks.getParameter<double>("xmax"));
	Cluster_Unkn_Endcap->setAxisTitle("Unknown TTCluster Stack", 1);
	Cluster_Unkn_Endcap->setAxisTitle("Number of Clusters", 2);
	
	HistoName = "Cluster_Comb_Endcap";
	Cluster_Comb_Endcap = dqmStore_->book1D(HistoName, HistoName,
																				psTTClusterStacks.getParameter<int32_t>("Nbinsx"),
																				psTTClusterStacks.getParameter<double>("xmin"),
																				psTTClusterStacks.getParameter<double>("xmax"));
	Cluster_Comb_Endcap->setAxisTitle("Combinatorial TTCluster Stack", 1);
	Cluster_Comb_Endcap->setAxisTitle("Number of Clusters", 2);
	
}//end of method


// ------------ method called once each job just after ending the event loop  ------------
void 
OuterTrackerMonitorCluster::endJob(void) 
{
	
}

DEFINE_FWK_MODULE(OuterTrackerMonitorCluster);

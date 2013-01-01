// -*- C++ -*-
//
// Package:     SiStripChannelChargeFilter
// Class  :     TECClusterFilter
// 
//
// Original Author: sfricke 


#include "EventFilter/SiStripChannelChargeFilter/interface/TECClusterFilter.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;

namespace cms
{

  TECClusterFilter::TECClusterFilter(const edm::ParameterSet& ps){
    //
    ModulesToBeExcluded.clear();
    ModulesToBeExcluded = ps.getParameter< std::vector<unsigned> >("ModulesToBeExcluded");
    edm::LogInfo("TECClusterFilter")<<"Clusters from "<<ModulesToBeExcluded.size()<<" modules will be ignored in the filter:";
    for( std::vector<uint32_t>::const_iterator imod = ModulesToBeExcluded.begin(); imod != ModulesToBeExcluded.end(); imod++){
      edm::LogInfo("TECClusterFilter")<< *imod;
    }
    //
    ChargeThresholdTEC=ps.getParameter<int>("ChargeThresholdTEC");
    edm::LogInfo("TECClusterFilter")<<"ChargeThresholdTEC"<<ChargeThresholdTEC;
    minNrOfTECClusters=ps.getParameter<int>("MinNrOfTECClusters");
    edm::LogInfo("TECClusterFilter")<<"MinNrOfTECClusters"<<minNrOfTECClusters;
    clusterProducer = ps.getParameter<string>("ClusterProducer");
    edm::LogInfo("TECClusterFilter")<<"ClusterProducer"<<clusterProducer;

    // also put decision in the event
    produces <int>();
  }

  bool TECClusterFilter::filter(edm::Event & e, edm::EventSetup const& c) 
  {
    edm::Handle< edm::DetSetVector<SiStripCluster> > h; //get SiStripCluster
    e.getByLabel(clusterProducer,h);
    bool decision=false;               // default value, only accept if set true in this loop
    unsigned int nr_clusters_above_threshold = 0;
    for (edm::DetSetVector<SiStripCluster>::const_iterator it=h->begin();it!=h->end();it++) 
      {
        DetId thedetId = DetId(it->detId());
	bool exclude_this_detid = false;
	for(vector<SiStripCluster>::const_iterator vit=(it->data).begin(); vit!=(it->data).end(); vit++) 
	  {
	    for( std::vector<uint32_t>::const_iterator imod = ModulesToBeExcluded.begin(); imod != ModulesToBeExcluded.end(); imod++ )
	      { if(*imod == thedetId.rawId()) exclude_this_detid = true;  } // found in exclusion list
	    if(  (! exclude_this_detid ) && (thedetId.subdetId()==StripSubdetector::TEC) ) // if not excluded and if TEC module
	      { // calculate sum of amplitudes
		unsigned int amplclus=0;
		// int amplclus=0;
		for(vector<uint8_t>::const_iterator ia=vit->amplitudes().begin(); ia!=vit->amplitudes().end(); ia++) 
		// for(vector<short>::const_iterator ia=vit->amplitudes().begin(); ia!=vit->amplitudes().end(); ia++) 
		  { if ((*ia)>0) amplclus+=(*ia); } // why should this be negative?
		if(amplclus>ChargeThresholdTEC) nr_clusters_above_threshold++;
	      }
	  }
      }
    if(nr_clusters_above_threshold>=minNrOfTECClusters) decision=true;
    std::auto_ptr< int > output_decision( new int(decision) );
    e.put(output_decision);
    return decision;
  }
}

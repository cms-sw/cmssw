// -*- C++ -*-
//
// Package:     SiStripChannelChargeFilter
// Class  :     ClusterMTCCFilter
// 
//
// Original Author:  dkcira


#include "EventFilter/SiStripChannelChargeFilter/interface/ClusterMTCCFilter.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;

namespace cms
{

ClusterMTCCFilter::ClusterMTCCFilter(const edm::ParameterSet& ps){
   //
   ModulesToBeExcluded.clear();
   ModulesToBeExcluded = ps.getParameter< std::vector<unsigned> >("ModulesToBeExcluded");
   edm::LogInfo("ClusterMTCCFilter")<<"Clusters from "<<ModulesToBeExcluded.size()<<" modules will be ignored in the filter:";
   for( std::vector<uint32_t>::const_iterator imod = ModulesToBeExcluded.begin(); imod != ModulesToBeExcluded.end(); imod++){
     edm::LogInfo("ClusterMTCCFilter")<< *imod;
   }
   //
   ChargeThresholdTIB=ps.getParameter<int>("ChargeThresholdTIB");
   ChargeThresholdTOB=ps.getParameter<int>("ChargeThresholdTOB");
   ChargeThresholdTEC=ps.getParameter<int>("ChargeThresholdTEC");
   MinClustersDiffComponents=ps.getParameter<int>("MinClustersDiffComponents");
   clusterProducer = ps.getParameter<string>("ClusterProducer");
   //
   produces <int>();
   produces <unsigned int >();
   produces < map<unsigned int,vector<SiStripCluster> > >();
}

bool ClusterMTCCFilter::filter(edm::Event & e, edm::EventSetup const& c) {

  edm::ESHandle<TrackerTopology> tTopo;
  c.get<IdealGeometryRecord>().get(tTopo);

  //get SiStripCluster
  edm::Handle< edm::DetSetVector<SiStripCluster> > h;
  e.getByLabel(clusterProducer,h);


  //
  unsigned int sum_of_cluster_charges=0;
  clusters_in_subcomponents.clear();
  // first find all clusters that are over the threshold
  for (edm::DetSetVector<SiStripCluster>::const_iterator it=h->begin();it!=h->end();it++) {
    for(vector<SiStripCluster>::const_iterator vit=(it->data).begin(); vit!=(it->data).end(); vit++){
      // calculate sum of amplitudes
      unsigned int amplclus=0;
      for(vector<uint8_t>::const_iterator ia=vit->amplitudes().begin(); ia!=vit->amplitudes().end(); ia++) {
        if ((*ia)>0) amplclus+=(*ia); // why should this be negative?
      }
      sum_of_cluster_charges += amplclus;
      DetId thedetId = DetId(it->detId());
      unsigned int generalized_layer = 0;
      bool exclude_this_detid = false;
      for( std::vector<uint32_t>::const_iterator imod = ModulesToBeExcluded.begin(); imod != ModulesToBeExcluded.end(); imod++ ){
          if(*imod == thedetId.rawId()) exclude_this_detid = true; // found in exclusion list
      }
      // apply different thresholds for TIB/TOB/TEC
      if( ! exclude_this_detid ){ // only consider if not in exclusion list
        if ( ( thedetId.subdetId()==StripSubdetector::TIB && amplclus>ChargeThresholdTIB )
          || ( thedetId.subdetId()==StripSubdetector::TOB && amplclus>ChargeThresholdTOB )
          || ( thedetId.subdetId()==StripSubdetector::TEC && amplclus>ChargeThresholdTEC )
          ){
          // calculate generalized_layer:  31 = TIB1, 32 = TIB2, 33 = TIB3, 50 = TOB, 60 = TEC
          if(thedetId.subdetId()==StripSubdetector::TIB){
             
             generalized_layer = 10*thedetId.subdetId() + tTopo->tibLayer(thedetId.rawId()) + tTopo->tibStereo(thedetId.rawId());
  	   if (tTopo->tibLayer(thedetId.rawId())==2){
  	     generalized_layer++;
  	     if (tTopo->tibGlued(thedetId.rawId())) edm::LogError("ClusterMTCCFilter")<<"WRONGGGG"<<endl;
  	   }
          }else{
            generalized_layer = 10*thedetId.subdetId();
  	  if(thedetId.subdetId()==StripSubdetector::TOB){
  	    
  	    generalized_layer += tTopo->tobLayer(thedetId.rawId());
  	  }
          }
          // fill clusters_in_subcomponents
          map<unsigned int,vector<SiStripCluster> >::iterator layer_it = clusters_in_subcomponents.find(generalized_layer);
          if(layer_it==clusters_in_subcomponents.end()){ // if layer not found yet, create DATA vector and generate map KEY + DATA
            vector<SiStripCluster> local_vector;
            local_vector.push_back(*vit);
            clusters_in_subcomponents.insert( std::make_pair( generalized_layer, local_vector) );
          }else{ // push into already existing vector
             (layer_it->second).push_back(*vit);
          }
        }
      }
    }
  }

  bool decision=false; // default value, only accept if set true in this loop
  unsigned int nr_of_subcomps_with_clusters=0;
// dk: 2006.08.24 - change filter decision as proposed by V. Ciulli. || TIB1 TIB2 counted as 1, TEC excluded
//  if( clusters_in_subcomponents[31].size()>0 ) nr_of_subcomps_with_clusters++; // TIB1
//  if( clusters_in_subcomponents[32].size()>0 ) nr_of_subcomps_with_clusters++; // TIB2
//  if( clusters_in_subcomponents[60].size()>0 ) nr_of_subcomps_with_clusters++; // TEC
  if( clusters_in_subcomponents[31].size()>0 ||  clusters_in_subcomponents[32].size()>0 ) nr_of_subcomps_with_clusters++; // TIB1 || TIB2
  if( clusters_in_subcomponents[33].size()>0 ) nr_of_subcomps_with_clusters++; // TIB3
  if( clusters_in_subcomponents[51].size()>0 ) nr_of_subcomps_with_clusters++; // TOB1
  if( clusters_in_subcomponents[52].size()>0 ) nr_of_subcomps_with_clusters++; // TOB2
  if(
     nr_of_subcomps_with_clusters >= MinClustersDiffComponents // more than 'MinClustersDiffComponents' components have at least 1 cluster
     ) {
      decision = true; // accept event
  }

  std::auto_ptr< int > output_decision( new int(decision) );
  e.put(output_decision);

  std::auto_ptr< unsigned int > output_sumofcharges( new unsigned int(sum_of_cluster_charges) );
  e.put(output_sumofcharges);

  std::auto_ptr< map<unsigned int,vector<SiStripCluster> > > output_clusters(new map<unsigned int,vector<SiStripCluster> > (clusters_in_subcomponents));
  e.put(output_clusters);

  return decision;
}
}

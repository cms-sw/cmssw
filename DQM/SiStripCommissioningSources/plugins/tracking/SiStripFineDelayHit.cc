// -*- C++ -*-
//
// Package:    SiStripFineDelayHit
// Class:      SiStripFineDelayHit
// 
/**\class SiStripFineDelayHit SiStripFineDelayHit.cc DQM/SiStripCommissioningSources/plugins/tracking/SiStripFineDelayHit.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Christophe DELAERE
//         Created:  Fri Nov 17 10:52:42 CET 2006
// $Id: SiStripFineDelayHit.cc,v 1.18 2013/02/27 15:07:58 davidlt Exp $
//
//


// system include files
#include <memory>
#include <utility>
#include <vector>
#include <algorithm>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiStripCluster/interface/SiStripClusterCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include <DataFormats/SiStripCommon/interface/SiStripEventSummary.h>
#include <DataFormats/SiStripCommon/interface/ConstantsForRunType.h>
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include <DataFormats/SiStripCommon/interface/SiStripFedKey.h>
#include <CondFormats/SiStripObjects/interface/FedChannelConnection.h>
#include <CondFormats/SiStripObjects/interface/SiStripFedCabling.h>
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include <CondFormats/SiStripObjects/interface/SiStripNoises.h>
#include <CondFormats/DataRecord/interface/SiStripFedCablingRcd.h>
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include <CondFormats/SiStripObjects/interface/SiStripNoises.h>


#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include <Geometry/CommonTopologies/interface/Topology.h>
#include <Geometry/CommonTopologies/interface/StripTopology.h>

#include <TrackingTools/PatternTools/interface/Trajectory.h>

#include "DQM/SiStripCommissioningSources/plugins/tracking/SiStripFineDelayHit.h"
#include "DQM/SiStripCommissioningSources/plugins/tracking/SiStripFineDelayTLA.h"
#include "DQM/SiStripCommissioningSources/plugins/tracking/SiStripFineDelayTOF.h"

#include "TMath.h"

//
// constructors and destructor
//
SiStripFineDelayHit::SiStripFineDelayHit(const edm::ParameterSet& iConfig):event_(0)
{
   //register your products
   produces<edm::DetSetVector<SiStripRawDigi> >("FineDelaySelection");
   //now do what ever other initialization is needed
   anglefinder_=new SiStripFineDelayTLA(iConfig);
   cosmic_ = iConfig.getParameter<bool>("cosmic");
   field_ = iConfig.getParameter<bool>("MagneticField");
   maxAngle_ = iConfig.getParameter<double>("MaxTrackAngle");
   minTrackP2_ = iConfig.getParameter<double>("MinTrackMomentum")*iConfig.getParameter<double>("MinTrackMomentum");
   maxClusterDistance_ = iConfig.getParameter<double>("MaxClusterDistance");
   clusterLabel_ = iConfig.getParameter<edm::InputTag>("ClustersLabel");
   trackLabel_ = iConfig.getParameter<edm::InputTag>("TracksLabel");
   seedLabel_  = iConfig.getParameter<edm::InputTag>("SeedsLabel");
   inputModuleLabel_ = iConfig.getParameter<edm::InputTag>( "InputModuleLabel" ) ;
   digiLabel_ = iConfig.getParameter<edm::InputTag>("DigiLabel");
   homeMadeClusters_ = iConfig.getParameter<bool>("NoClustering");
   explorationWindow_ = iConfig.getParameter<uint32_t>("ExplorationWindow");
   noTracking_ = iConfig.getParameter<bool>("NoTracking");
   mode_=0;
}

SiStripFineDelayHit::~SiStripFineDelayHit()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
   delete anglefinder_;
}

//
// member functions
//
std::pair<uint32_t, uint32_t> SiStripFineDelayHit::deviceMask(const StripSubdetector::SubDetector subdet,const int substructure, const TrackerTopology *tTopo)
{
  uint32_t rootDetId = 0;
  uint32_t maskDetId = 0;
  switch(subdet){
    case (int)StripSubdetector::TIB :
    {
      rootDetId = tTopo->tibDetId(substructure,0,0,0,0,0).rawId();
      maskDetId = tTopo->tibDetId(15,0,0,0,0,0).rawId();
      break;
    }
    case (int)StripSubdetector::TID :
    {
      rootDetId = tTopo->tidDetId(substructure>0 ? 2 : 1,abs(substructure),0,0,0,0).rawId();
      maskDetId = tTopo->tidDetId(3,15,0,0,0,0).rawId();
      break;
    }
    case (int)StripSubdetector::TOB :
    {
      rootDetId = tTopo->tobDetId(substructure,0,0,0,0).rawId();
      maskDetId = tTopo->tobDetId(15,0,0,0,0).rawId();
      break;
    }
    case (int)StripSubdetector::TEC :
    {
      rootDetId = tTopo->tecDetId(substructure>0 ? 2 : 1,abs(substructure),0,0,0,0,0).rawId();
      maskDetId = tTopo->tecDetId(3,15,0,0,0,0,0).rawId();
      break;
    }
  }
  return std::make_pair(maskDetId,rootDetId);
}

std::vector< std::pair<uint32_t,std::pair<double, double> > > SiStripFineDelayHit::detId(const TrackerGeometry& tracker,const reco::Track* tk, const std::vector<Trajectory>& trajVec, const TrackerTopology *tTopo, const StripSubdetector::SubDetector subdet,const int substructure)
{
  if(substructure==0xff) return detId(tracker,tk,trajVec,0,0,tTopo);
  // first determine the root detId we are looking for
  std::pair<uint32_t, uint32_t> mask = deviceMask(subdet,substructure,tTopo);
  // then call the method that loops on recHits
  return detId(tracker,tk,trajVec,mask.first,mask.second,tTopo);
}

std::vector< std::pair<uint32_t,std::pair<double, double> > > SiStripFineDelayHit::detId(const TrackerGeometry& tracker,const reco::Track* tk, const std::vector<Trajectory>& trajVec, const uint32_t& maskDetId, const uint32_t& rootDetId, const TrackerTopology *tTopo)
{
  bool onDisk = ((maskDetId==tTopo->tidDetId(3,15,0,0,0,0).rawId())||(maskDetId==tTopo->tecDetId(3,15,0,0,0,0,0).rawId())) ;
  std::vector< std::pair<uint32_t,std::pair<double, double> > > result;
  std::vector<uint32_t> usedDetids;
  // now loop on recHits to find the right detId plus the track local angle
  std::vector<std::pair< std::pair<DetId, LocalPoint> ,float> > hitangle;
  if(!cosmic_) {
    // use trajectories in event.
    // we have first to find the right trajectory for the considered track.
    for(std::vector<Trajectory>::const_iterator traj = trajVec.begin(); traj< trajVec.end(); ++traj) {
      if(
        ((traj->lastMeasurement().recHit()->geographicalId().rawId() == (*(tk->recHitsEnd()-1))->geographicalId().rawId()) &&
        ( traj->lastMeasurement().recHit()->localPosition().x() == (*(tk->recHitsEnd()-1))->localPosition().x())               ) ||
        ((traj->firstMeasurement().recHit()->geographicalId().rawId() == (*(tk->recHitsEnd()-1))->geographicalId().rawId()) &&
        ( traj->firstMeasurement().recHit()->localPosition().x() == (*(tk->recHitsEnd()-1))->localPosition().x())              )   ) {
          hitangle = anglefinder_->findtrackangle(*traj);
          break;
      }
    }
  } else {
    edm::Handle<TrajectorySeedCollection> seedcoll;
    event_->getByLabel(seedLabel_,seedcoll);
    // use trajectories in event.
    hitangle = anglefinder_->findtrackangle(trajVec);
  }
  LogDebug("DetId") << "number of hits for the track: " << hitangle.size();
  std::vector<std::pair< std::pair<DetId, LocalPoint> ,float> >::iterator iter;
  // select the interesting DetIds, based on the ID and TLA
  for(iter=hitangle.begin();iter!=hitangle.end();iter++){
    // check the detId.
    // if substructure was 0xff, then maskDetId and rootDetId == 0 
    // this implies all detids are accepted. (also if maskDetId=rootDetId=0 explicitely).
    // That "unusual" mode of operation allows to analyze also Latency scans
    LogDebug("DetId") << "check the detid: " << std::hex << (iter->first.first.rawId()) << " vs " << rootDetId
                      << " with a mask of "  << maskDetId << std::dec << std::endl;

    if(((iter->first.first.rawId() & maskDetId) != rootDetId)) continue;
    if(std::find(usedDetids.begin(),usedDetids.end(),iter->first.first.rawId())!=usedDetids.end()) continue;
    // check the local angle (extended to the equivalent angle correction)
    LogDebug("DetId") << "check the angle: " << fabs((iter->second));
    if(1-fabs(fabs(iter->second)-1)<cos(maxAngle_/180.*TMath::Pi())) continue;
    // returns the detid + the time of flight to there
    std::pair<uint32_t,std::pair<double, double> > el;
    std::pair<double, double> subel;
    el.first = iter->first.first.rawId();
    // here, we compute the TOF.
    // For cosmics, some track parameters are missing. Parameters are recomputed.
    // for our calculation, the track momemtum at any point is enough:
    // only used without B field or for the sign of Pz.
    double trackParameters[5];
    for(int i=0;i<5;i++) trackParameters[i] = tk->parameters()[i];
    if(cosmic_) SiStripFineDelayTOF::trackParameters(*tk,trackParameters);
    double hit[3];
    const GeomDetUnit* det(tracker.idToDetUnit(iter->first.first));
    Surface::GlobalPoint gp = det->surface().toGlobal(iter->first.second);
    hit[0]=gp.x();
    hit[1]=gp.y();
    hit[2]=gp.z();
    double phit[3];
    phit[0] = tk->momentum().x();
    phit[1] = tk->momentum().y();
    phit[2] = tk->momentum().z();
    subel.first = SiStripFineDelayTOF::timeOfFlight(cosmic_,field_,trackParameters,hit,phit,onDisk);
    subel.second = iter->second;
    el.second = subel;
    // returns the detid + TOF
    result.push_back(el);
    usedDetids.push_back(el.first);
  }
  return result;
}

bool SiStripFineDelayHit::rechit(reco::Track* tk,uint32_t det_id)
{
  for(trackingRecHit_iterator it = tk->recHitsBegin(); it != tk->recHitsEnd(); it++) 
    if((*it)->geographicalId().rawId() == det_id) {
      return (*it)->isValid();
      break;
    }
  return false;
}

// VI January 2012: FIXME
// do not understand what is going on here: each hit has a cluster: by definition will be the closest!
std::pair<const SiStripCluster*,double> SiStripFineDelayHit::closestCluster(const TrackerGeometry& tracker,const reco::Track* tk,const uint32_t& det_id ,const edmNew::DetSetVector<SiStripCluster>& clusters, const edm::DetSetVector<SiStripDigi>& hits)
{
  std::pair<const SiStripCluster*,double> result(NULL,0.);
  double hitStrip = -1;
  int nstrips = -1;
  // localize the crossing point of the track on the module
  for(trackingRecHit_iterator it = tk->recHitsBegin(); it != tk->recHitsEnd(); it++) {
    LogDebug("closestCluster") << "(*it)->geographicalId().rawId() vs det_id" << (*it)->geographicalId().rawId() << " " <<  det_id;
    //handle the mono rechits
    if((*it)->geographicalId().rawId() == det_id) {
      if(!(*it)->isValid()) continue;
      LogDebug("closestCluster") << " using the single mono hit";
      LocalPoint lp = (*it)->localPosition();
      const GeomDetUnit* gdu = static_cast<const GeomDetUnit*>(tracker.idToDet((*it)->geographicalId()));
      MeasurementPoint p = gdu->topology().measurementPosition(lp);
      hitStrip = p.x();
      nstrips = (dynamic_cast<const StripTopology*>(&(gdu->topology())))->nstrips();
      break;
    }
    /* FIXME: local position is not there anymore...
    //handle stereo part of matched hits
    //one could try to cast to SiStripMatchedRecHit2D but it is faster to look at the detid
    else if((det_id - (*it)->geographicalId().rawId())==1) {
      const SiStripMatchedRecHit2D* hit2D = dynamic_cast<const SiStripMatchedRecHit2D*>(&(**it));
      if(!hit2D) continue; // this is a security that should never trigger
      const SiStripRecHit2D* stereo = hit2D->stereoHit();
      if(!stereo) continue; // this is a security that should never trigger
      if(!stereo->isValid()) continue;
      LogDebug("closestCluster") << " using the stereo hit";
      LocalPoint lp = stereo->localPosition();
      const GeomDetUnit* gdu = static_cast<const GeomDetUnit*>(tracker.idToDet(stereo->geographicalId()));
      MeasurementPoint p = gdu->topology().measurementPosition(lp);
      hitStrip = p.x();
      nstrips = (dynamic_cast<const StripTopology*>(&(gdu->topology())))->nstrips();
      break;
    }
    //handle mono part of matched hits
    //one could try to cast to SiStripMatchedRecHit2D but it is faster to look at the detid
    else if((det_id - (*it)->geographicalId().rawId())==2) {
      const SiStripMatchedRecHit2D* hit2D = dynamic_cast<const SiStripMatchedRecHit2D*>(&(**it));
      if(!hit2D) continue; // this is a security that should never trigger
      const SiStripRecHit2D* mono = hit2D->monoHit();
      if(!mono) continue; // this is a security that should never trigger
      if(!mono->isValid()) continue;
      LogDebug("closestCluster") << " using the mono hit";
      LocalPoint lp = mono->localPosition();
      const GeomDetUnit* gdu = static_cast<const GeomDetUnit*>(tracker.idToDet(mono->geographicalId()));
      MeasurementPoint p = gdu->topology().measurementPosition(lp);
      hitStrip = p.x();
      nstrips = (dynamic_cast<const StripTopology*>(&(gdu->topology())))->nstrips();
      break;
    }
    */
  }
  LogDebug("closestCluster") << " hit strip = " << hitStrip;
  if(hitStrip<0) return result;
  if(homeMadeClusters_) {
    // take the list of digis on the module
    for (edm::DetSetVector<SiStripDigi>::const_iterator DSViter=hits.begin(); DSViter!=hits.end();DSViter++){
      if(DSViter->id==det_id)  {
        // loop from hitstrip-n to hitstrip+n (explorationWindow_) and select the highest strip
	int minStrip = int(round(hitStrip))- explorationWindow_;
	minStrip = minStrip<0 ? 0 : minStrip;
	int maxStrip = int(round(hitStrip)) + explorationWindow_ + 1;
	maxStrip = maxStrip>=nstrips ? nstrips-1 : maxStrip;
	edm::DetSet<SiStripDigi>::const_iterator rangeStart = DSViter->end();
	edm::DetSet<SiStripDigi>::const_iterator rangeStop  = DSViter->end();
	for(edm::DetSet<SiStripDigi>::const_iterator digiIt = DSViter->begin(); digiIt!=DSViter->end(); ++digiIt) {
	  if(digiIt->strip()>=minStrip && rangeStart == DSViter->end()) rangeStart = digiIt;
	  if(digiIt->strip()<=maxStrip) rangeStop = digiIt;
	}
	if(rangeStart != DSViter->end()) {
	  if(rangeStop !=DSViter->end()) ++rangeStop;
          // build a fake cluster 
          LogDebug("closestCluster") << "build a fake cluster ";
          SiStripCluster* newCluster = new SiStripCluster(det_id,SiStripCluster::SiStripDigiRange(rangeStart,rangeStop)); // /!\ ownership transfered
          result.first = newCluster;
          result.second = fabs(newCluster->barycenter()-hitStrip);
	}
	break;
      }
    }
  } else {
  // loop on the detsetvector<cluster> to find the right one
   for (edmNew::DetSetVector<SiStripCluster>::const_iterator DSViter=clusters.begin(); DSViter!=clusters.end();DSViter++ ) 
     if(DSViter->id()==det_id)  {
        LogDebug("closestCluster") << " detset with the right detid. ";
        edmNew::DetSet<SiStripCluster>::const_iterator begin=DSViter->begin();
        edmNew::DetSet<SiStripCluster>::const_iterator end  =DSViter->end();
	//find the cluster close to the hitStrip
	result.second = 1000.;
        for(edmNew::DetSet<SiStripCluster>::const_iterator iter=begin;iter!=end;++iter) {
	  double dist = fabs(iter->barycenter()-hitStrip);
	  if(dist<result.second) { result.second = dist; result.first = &(*iter); }
	}
        break;
     }
  }
  return result;
}

// ------------ method called to produce the data  ------------
void
SiStripFineDelayHit::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   //Retrieve tracker topology from geometry
   edm::ESHandle<TrackerTopology> tTopoHandle;
   iSetup.get<IdealGeometryRecord>().get(tTopoHandle);
   const TrackerTopology* const tTopo = tTopoHandle.product(); 
   
   // Retrieve commissioning information from "event summary"
   edm::Handle<SiStripEventSummary> runsummary;
   iEvent.getByLabel( inputModuleLabel_, runsummary );
   if(runsummary->runType()==sistrip::APV_LATENCY) mode_ = 2; // LatencyScan
   else if(runsummary->runType()==sistrip::FINE_DELAY) mode_ = 1; // DelayScan
   else { 
    mode_ = 0; //unknown
    return;
   }

   if(noTracking_) {
      produceNoTracking(iEvent,iSetup);
      return;
   }
   event_ = &iEvent;
   // container for the selected hits
   std::vector< edm::DetSet<SiStripRawDigi> > output;
   output.reserve(100);
   // access the tracks
   edm::Handle<reco::TrackCollection> trackCollection;
   iEvent.getByLabel(trackLabel_,trackCollection);  
   const reco::TrackCollection *tracks=trackCollection.product();
   edm::ESHandle<TrackerGeometry> tracker;
   iSetup.get<TrackerDigiGeometryRecord>().get(tracker);
   if (tracks->size()) {
     anglefinder_->init(iEvent,iSetup);
     LogDebug("produce") << "Found " << tracks->size() << " tracks.";
     // look at the hits if one needs them
     edm::Handle< edm::DetSetVector<SiStripDigi> > hits;
     const edm::DetSetVector<SiStripDigi>* hitSet = NULL;
     if(homeMadeClusters_) {
       iEvent.getByLabel(digiLabel_,hits);
       hitSet = hits.product();
     }
     // look at the clusters 
     edm::Handle<edmNew::DetSetVector<SiStripCluster> > clusters;
     iEvent.getByLabel(clusterLabel_, clusters);
     const edmNew::DetSetVector<SiStripCluster>* clusterSet = clusters.product();
     // look at the trajectories if they are in the event
     std::vector<Trajectory> trajVec;
     edm::Handle<std::vector<Trajectory> > TrajectoryCollection;
     iEvent.getByLabel(trackLabel_,TrajectoryCollection);
     trajVec = *(TrajectoryCollection.product());
     // loop on tracks
     for(reco::TrackCollection::const_iterator itrack = tracks->begin(); itrack<tracks->end(); itrack++) {
       // first check the track Pt
       if((itrack->px()*itrack->px()+itrack->py()*itrack->py()+itrack->pz()*itrack->pz())<minTrackP2_) continue;
       // check that we have something in the layer we are interested in
       std::vector< std::pair<uint32_t,std::pair<double,double> > > intersections;
       if(mode_==1) {
         // Retrieve and decode commissioning information from "event summary"
         edm::Handle<SiStripEventSummary> summary;
         iEvent.getByLabel( inputModuleLabel_, summary );
         uint32_t layerCode = (const_cast<SiStripEventSummary*>(summary.product())->layerScanned())>>16;
         StripSubdetector::SubDetector subdet = StripSubdetector::TIB;
         if(((layerCode>>6)&0x3)==0) subdet = StripSubdetector::TIB;
         else if(((layerCode>>6)&0x3)==1) subdet = StripSubdetector::TOB;
         else if(((layerCode>>6)&0x3)==2) subdet = StripSubdetector::TID;
         else if(((layerCode>>6)&0x3)==3) subdet = StripSubdetector::TEC;
         int32_t layerIdx = (layerCode&0xF)*(((layerCode>>4)&0x3) ? -1 : 1);
         intersections = detId(*tracker,&(*itrack),trajVec,tTopo,subdet,layerIdx);
       } else {
         // for latency scans, no layer is specified -> no cut on detid
         intersections = detId(*tracker,&(*itrack),trajVec,tTopo);
       }
       LogDebug("produce") << "  Found " << intersections.size() << " interesting intersections." << std::endl;
       for(std::vector< std::pair<uint32_t,std::pair<double,double> > >::iterator it = intersections.begin();it<intersections.end();it++) {
         std::pair<const SiStripCluster*,double> candidateCluster = closestCluster(*tracker,&(*itrack),it->first,*clusterSet,*hitSet);
         if(candidateCluster.first) {
           LogDebug("produce") << "    Found a cluster."<< std::endl;
           // cut on the distance 
  	 if(candidateCluster.second>maxClusterDistance_) continue; 
           LogDebug("produce") << "    The cluster is close enough."<< std::endl;
  	 // build the rawdigi corresponding to the leading strip and save it
  	 // here, only the leading strip is retained. All other rawdigis in the module are set to 0.
  	 const std::vector< uint8_t >& amplitudes = candidateCluster.first->amplitudes();
  	 uint8_t leadingCharge = 0;
  	 uint8_t leadingStrip = candidateCluster.first->firstStrip();
  	 uint8_t leadingPosition = 0;
  	 for(std::vector< uint8_t >::const_iterator amplit = amplitudes.begin();amplit<amplitudes.end();amplit++,leadingStrip++) {
  	   if(leadingCharge<*amplit) {
  	     leadingCharge = *amplit;
  	     leadingPosition = leadingStrip;
  	   }
  	 }

         // look for an existing detset
	 std::vector< edm::DetSet<SiStripRawDigi> >::iterator newdsit = output.begin();
	 for(;newdsit!=output.end()&&newdsit->detId()!=connectionMap_[it->first];++newdsit) {}
	 // if there is no detset yet, create it.
         if(newdsit==output.end()) {
	   edm::DetSet<SiStripRawDigi> newds(connectionMap_[it->first]);
	   output.push_back(newds);
	   newdsit = output.end()-1;
	 }

         LogDebug("produce") << " New Hit...   TOF:" << it->second.first << ", charge: " << int(leadingCharge) 
                             << " at " << int(leadingPosition) << "." << std::endl
                             << "Angular correction: " << it->second.second 
                             << " giving a final value of " << int(leadingCharge*fabs(it->second.second)) 
			     << " for fed key = " << connectionMap_[it->first] << " (detid=" << it->first << ")" ;
  	 // apply corrections to the leading charge, but only if it has not saturated.
  	 if(leadingCharge<255) {
  	   // correct the leading charge for the crossing angle
    	   leadingCharge = uint8_t(leadingCharge*fabs(it->second.second));
           // correct for module thickness for TEC and TOB
	   if((((it->first>>25)&0x7f)==0xd) ||
	      ((((it->first>>25)&0x7f)==0xe) && (((it->first>>5)&0x7)>4)))
             leadingCharge = uint8_t((leadingCharge*0.64));
  	 }
  	 //code the time of flight in the digi
  	 unsigned int tof = abs(int(round(it->second.first*10)));
  	 tof = tof>255 ? 255 : tof;
  	 SiStripRawDigi newSiStrip(leadingCharge + (tof<<8));
  	 newdsit->push_back(newSiStrip);
  	 LogDebug("produce") << "New edm::DetSet<SiStripRawDigi> added.";
         }
         if(homeMadeClusters_) delete candidateCluster.first; // we are owner of home-made clusters
       }
     }
   }
   // add the selected hits to the event.
   LogDebug("produce") << "Putting " << output.size() << " new hits in the event.";
   std::auto_ptr< edm::DetSetVector<SiStripRawDigi> > formatedOutput(new edm::DetSetVector<SiStripRawDigi>(output) );
   iEvent.put(formatedOutput,"FineDelaySelection");
}

// Simple solution when tracking is not available/ not working
void
SiStripFineDelayHit::produceNoTracking(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    //Retrieve tracker topology from geometry
   edm::ESHandle<TrackerTopology> tTopoHandle;
   iSetup.get<IdealGeometryRecord>().get(tTopoHandle);
   const TrackerTopology* const tTopo = tTopoHandle.product(); 
   
   event_ = &iEvent;
   // container for the selected hits
   std::vector< edm::DetSet<SiStripRawDigi> > output;
   output.reserve(100);
   // Retrieve and decode commissioning information from "event summary"
   edm::Handle<SiStripEventSummary> summary;
   iEvent.getByLabel( inputModuleLabel_, summary );
   uint32_t layerCode = (const_cast<SiStripEventSummary*>(summary.product())->layerScanned())>>16;
   StripSubdetector::SubDetector subdet = StripSubdetector::TIB;
   if(((layerCode>>6)&0x3)==0) subdet = StripSubdetector::TIB;
   else if(((layerCode>>6)&0x3)==1) subdet = StripSubdetector::TOB;
   else if(((layerCode>>6)&0x3)==2) subdet = StripSubdetector::TID;
   else if(((layerCode>>6)&0x3)==3) subdet = StripSubdetector::TEC;
   int32_t layerIdx = (layerCode&0xF)*(((layerCode>>4)&0x3) ? -1 : 1);
   std::pair<uint32_t, uint32_t> mask = deviceMask(subdet,layerIdx,tTopo);
   // look at the clusters 
   edm::Handle<edmNew::DetSetVector<SiStripCluster> > clusters;
   iEvent.getByLabel(clusterLabel_,clusters);
   for (edmNew::DetSetVector<SiStripCluster>::const_iterator DSViter=clusters->begin(); DSViter!=clusters->end();DSViter++ ) {
     // check that we are in the layer of interest
     if(mode_==1 && ((DSViter->id() & mask.first) != mask.second) ) continue;
     // iterate over clusters
     edmNew::DetSet<SiStripCluster>::const_iterator begin=DSViter->begin();
     edmNew::DetSet<SiStripCluster>::const_iterator end  =DSViter->end();
     edm::DetSet<SiStripRawDigi> newds(connectionMap_[DSViter->id()]);
     for(edmNew::DetSet<SiStripCluster>::const_iterator iter=begin;iter!=end;++iter) {
         // build the rawdigi corresponding to the leading strip and save it
         // here, only the leading strip is retained. All other rawdigis in the module are set to 0.
	 const std::vector< uint8_t >& amplitudes = iter->amplitudes();
	 uint8_t leadingCharge = 0;
	 uint8_t leadingStrip = iter->firstStrip();
	 uint8_t leadingPosition = 0;
	 for(std::vector< uint8_t >::const_iterator amplit = amplitudes.begin();amplit<amplitudes.end();amplit++,leadingStrip++) {
	   if(leadingCharge<*amplit) {
	     leadingCharge = *amplit;
	     leadingPosition = leadingStrip;
	   }
	 }
         // apply some sanity cuts. This is needed since we don't use tracking to clean clusters
         // 1.5< noise <8
         // charge<250
         // 50 > s/n > 10
         edm::ESHandle<SiStripNoises> noiseHandle_;
         iSetup.get<SiStripNoisesRcd>().get(noiseHandle_);
         SiStripNoises::Range detNoiseRange = noiseHandle_->getRange(DSViter->id());  
         float noise=noiseHandle_->getNoise(leadingPosition, detNoiseRange);   
         if( noise<1.5 ) continue;
         if( leadingCharge>=250 || noise>=8 || leadingCharge/noise>50 || leadingCharge/noise<10 ) continue;
	 // apply some correction to the leading charge, but only if it has not saturated.
	 if(leadingCharge<255) {
	   // correct for modulethickness for TEC and TOB
	   if((((((DSViter->id())>>25)&0x7f)==0xd)||((((DSViter->id())>>25)&0x7f)==0xe))&&((((DSViter->id())>>5)&0x7)>4)) 
	      leadingCharge = uint8_t((leadingCharge*0.64));
	 }
	 //code the time of flight == 0 in the digi
	 SiStripRawDigi newSiStrip(leadingCharge);
	 newds.push_back(newSiStrip);
     }
     //store into the detsetvector
     output.push_back(newds);
     LogDebug("produce") << "New edm::DetSet<SiStripRawDigi> added with fedkey = " 
                         << std::hex << std::setfill('0') << std::setw(8) 
                         << connectionMap_[DSViter->id()] << std::dec;
   }
   // add the selected hits to the event.
   LogDebug("produce") << "Putting " << output.size() << " new hits in the event.";
   std::auto_ptr< edm::DetSetVector<SiStripRawDigi> > formatedOutput(new edm::DetSetVector<SiStripRawDigi>(output) );
   iEvent.put(formatedOutput,"FineDelaySelection");
}

// ------------ method called once each job just before starting event loop  ------------
void 
SiStripFineDelayHit::beginRun(const edm::Run & run, const edm::EventSetup & iSetup)
{
   // Retrieve FED cabling object
   edm::ESHandle<SiStripFedCabling> cabling;
   iSetup.get<SiStripFedCablingRcd>().get( cabling );
   const std::vector< uint16_t > & feds = cabling->feds() ;
   for(std::vector< uint16_t >::const_iterator fedid = feds.begin();fedid<feds.end();++fedid) {
     const std::vector< FedChannelConnection > & connections = cabling->connections(*fedid);
     for(std::vector< FedChannelConnection >::const_iterator conn=connections.begin();conn<connections.end();++conn) {
     /*
       SiStripFedKey key(conn->fedId(),
                         SiStripFedKey::feUnit(conn->fedCh()),
			 SiStripFedKey::feChan(conn->fedCh()));
       connectionMap_[conn->detId()] = key.key();
     */
     // the key is computed using an alternate formula for performance reasons.
     connectionMap_[conn->detId()] = ( ( conn->fedId() & sistrip::invalid_ ) << 16 ) | ( conn->fedCh() & sistrip::invalid_ );
     }
   }
}

#include <memory>
#include "FastSimulation/Tracking/interface/PixelTracksMaker.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2DCollection.h" 
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "TrackingTools/TransientTrackingRecHit/interface/GenericTransientTrackingRecHit.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"

#include "FastSimulation/BaseParticlePropagator/interface/BaseParticlePropagator.h"
#include "FastSimulation/ParticlePropagator/interface/ParticlePropagator.h"
//
//Pixel Specific stuff
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducerFactory.h"

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitter.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitterFactory.h"

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackFilter.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackFilterFactory.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/Common/interface/OrphanHandle.h"
#include <vector>

using namespace pixeltrackfitting;
using edm::ParameterSet;



PixelTracksMaker::PixelTracksMaker(const edm::ParameterSet& conf) 
  : theConfig(conf), theFitter(0), theFilter(0), theRegionProducer(0)
{  
  produces<reco::TrackCollection>();
  produces<TrackingRecHitCollection>();
  produces<reco::TrackExtraCollection>();

  // The smallest true pT for a track candidate
  pTMin = conf.getParameter<double>("pTMin");
  pTMin *= pTMin;  // Cut is done of perp2() - CPU saver
  
  // The smallest number of Rec Hits for a track candidate
  minRecHits = conf.getParameter<unsigned int>("MinRecHits");

  // The smallest true impact parameters (d0 and z0) for a track candidate
  maxD0 = conf.getParameter<double>("MaxD0");
  maxZ0 = conf.getParameter<double>("MaxZ0");

  // The name of the hit producer
  hitProducer = conf.getParameter<std::string>("HitProducer");

  // The cuts for PixelSeeding (doublets) 
  originRadius = conf.getParameter<double>("originRadius");
  originHalfLength = conf.getParameter<double>("originHalfLength");
  originpTMin = conf.getParameter<double>("originpTMin");

}

  
// Virtual destructor needed.
PixelTracksMaker::~PixelTracksMaker() {

  delete theFilter;
  delete theFitter;
  delete theRegionProducer;

} 
 

void PixelTracksMaker::beginJob(const edm::EventSetup& es)
{

  ParameterSet regfactoryPSet = theConfig.getParameter<ParameterSet>("RegionFactoryPSet");
  std::string regfactoryName = regfactoryPSet.getParameter<std::string>("ComponentName");
  theRegionProducer = TrackingRegionProducerFactory::get()->create(regfactoryName,regfactoryPSet);
  
  ParameterSet fitterPSet = theConfig.getParameter<ParameterSet>("FitterPSet");
  std::string fitterName = fitterPSet.getParameter<std::string>("ComponentName");
  theFitter = PixelFitterFactory::get()->create( fitterName, fitterPSet);
  
  ParameterSet filterPSet = theConfig.getParameter<ParameterSet>("FilterPSet");
  std::string  filterName = filterPSet.getParameter<std::string>("ComponentName");
  theFilter = PixelTrackFilterFactory::get()->create( filterName, filterPSet);
  
  edm::ESHandle<MagneticField>          magField;
  edm::ESHandle<TrackerGeometry>        geometry;
  
  es.get<IdealMagneticFieldRecord>().get(magField);
  es.get<TrackerDigiGeometryRecord>().get(geometry);
  
  theMagField = &(*magField);
  theGeometry = &(*geometry);
  
  
}

// Functions that gets called by framework every event
void 
PixelTracksMaker::produce(edm::Event& e, const edm::EventSetup& es) {        
  
  unsigned nSimTracks = 0;
  unsigned nTracksWithHits = 0;
  unsigned nTracksWithPT = 0;
  unsigned nTracksWithD0Z0 = 0;
  unsigned nTriplets = 0;
  unsigned nFilterTracks = 0;
  unsigned nPixelTracks = 0;
  // unsigned nCleanedTracks = 0;
  
  std::auto_ptr<reco::TrackCollection> tracks(new reco::TrackCollection);    
  std::auto_ptr<TrackingRecHitCollection> recHits(new TrackingRecHitCollection);
  std::auto_ptr<reco::TrackExtraCollection> trackExtras(new reco::TrackExtraCollection);
  typedef std::vector<const TrackingRecHit *> RecHits;
  
  TracksWithRecHits pixeltracks;
  TracksWithRecHits cleanedTracks;
  
  edm::Handle<edm::SimTrackContainer> theSimTracks;
  e.getByLabel("famosSimHits",theSimTracks);
  
  edm::Handle<edm::SimVertexContainer> theSimVtx;
  e.getByLabel("famosSimHits",theSimVtx);
  
  edm::Handle<SiTrackerGSRecHit2DCollection> theGSRecHits;
  e.getByLabel(hitProducer, theGSRecHits);
  
  // No tracking attempted if no hits (but put an empty collection in the event)!
  if(theGSRecHits->size() == 0) {
    e.put(tracks);
    return;
  }
  
  //only one region Global, but it is called at every event...
  //maybe there is a smarter way to set it only once
  //NEED TO FIX
  typedef std::vector<TrackingRegion* > Regions;
  typedef Regions::const_iterator IR;
  Regions regions = theRegionProducer->regions(e,es);
  for (IR ir=regions.begin(), irEnd=regions.end(); ir < irEnd; ++ir) {
    const TrackingRegion & region = **ir;
    
    // The vector of simTrack Id's carrying GSRecHits
    const std::vector<unsigned> theSimTrackIds = theGSRecHits->ids();
    
    // loop over SimTrack Id's
    for ( unsigned tkId=0;  tkId != theSimTrackIds.size(); ++tkId ) {
      
      ++nSimTracks;
      unsigned simTrackId = theSimTrackIds[tkId];
      const SimTrack& theSimTrack = (*theSimTracks)[simTrackId]; 
      
      SiTrackerGSRecHit2DCollection::range theRecHitRange = theGSRecHits->get(simTrackId);
      SiTrackerGSRecHit2DCollection::const_iterator theRecHitRangeIteratorBegin = theRecHitRange.first;
      SiTrackerGSRecHit2DCollection::const_iterator theRecHitRangeIteratorEnd   = theRecHitRange.second;
      SiTrackerGSRecHit2DCollection::const_iterator iterRecHit;
      SiTrackerGSRecHit2DCollection::const_iterator iterRecHit2;
      SiTrackerGSRecHit2DCollection::const_iterator iterRecHit3;
      
      // Request a minimum number of RecHits
      unsigned numberOfRecHits = 0;
      for ( iterRecHit = theRecHitRangeIteratorBegin; 
	    iterRecHit != theRecHitRangeIteratorEnd; 
	    ++iterRecHit) ++numberOfRecHits;
      if ( numberOfRecHits < minRecHits ) continue;
      ++nTracksWithHits;
      
      // Request a minimum pT for the sim track
      if ( theSimTrack.momentum().perp2() < pTMin ) continue;
      ++nTracksWithPT;
      
      // Check that the sim track comes from the main vertex (loose cut)
      int vertexIndex = theSimTrack.vertIndex();
      const SimVertex& theSimVertex = (*theSimVtx)[vertexIndex]; 
      
      BaseParticlePropagator theParticle = 
	BaseParticlePropagator( 
			       RawParticle(XYZTLorentzVector(theSimTrack.momentum().px(),
							     theSimTrack.momentum().py(),
							     theSimTrack.momentum().pz(),
							     theSimTrack.momentum().e()),
					   XYZTLorentzVector(theSimVertex.position().x(),
							     theSimVertex.position().y(),
							     theSimVertex.position().z(),
							     theSimVertex.position().t())),
			       0.,0.,4.);
      theParticle.setCharge((*theSimTracks)[simTrackId].charge());
      if ( theParticle.xyImpactParameter() > maxD0 ) continue;
      if ( fabs( theParticle.zImpactParameter() ) > maxZ0 ) continue;
      ++nTracksWithD0Z0;
      
      std::vector<const TrackingRecHit*> TripletHits;
      
      bool compatible = false;  //doublet should satisfy the PixelSeeding cuts
      bool triplet = false;     //triplet = if there is a third hit in the expected detector
      const SiTrackerGSRecHit2D *hit1;
      const SiTrackerGSRecHit2D *hit2;
      const SiTrackerGSRecHit2D *hit3;
      
      //first triplet type: lay1+lay2+(lay3 || disk1)
      for ( iterRecHit = theRecHitRangeIteratorBegin; iterRecHit != theRecHitRangeIteratorEnd; ++iterRecHit) {
	hit1 = &(*iterRecHit);
	//first hit always on innermost layer on barrel
	if((unsigned int)(hit1->geographicalId().subdetId())== PixelSubdetector::PixelBarrel){
	  const DetId& detId = hit1->geographicalId();
	  PXBDetId pxbid1(detId.rawId()); 
	  unsigned int layerNumberHit1 = pxbid1.layer();  
	  if(layerNumberHit1 == 1){
	    const GeomDet* geomDet( theGeometry->idToDet(detId) );
	    GlobalPoint gpos1 = geomDet->surface().toGlobal(hit1->localPosition());
	    
	    for ( iterRecHit2 = iterRecHit+1; iterRecHit2 != theRecHitRangeIteratorEnd; ++iterRecHit2) {
	      hit2 = &(*iterRecHit2);
	      //second hit in the barrel 
	      if((unsigned int)hit2->geographicalId().subdetId()== PixelSubdetector::PixelBarrel) {
		const DetId& detId = hit2->geographicalId();
		PXBDetId pxbid2(detId.rawId()); 
		unsigned int layerNumberHit2 = pxbid2.layer();  
		if(layerNumberHit2 == 2){
		  const GeomDet* geomDet( theGeometry->idToDet(detId) );
		  GlobalPoint gpos2 = geomDet->surface().toGlobal(hit2->localPosition());
		  //use the same requirement as used for PixelSeeding
		  compatible = compatibleWithVertex(gpos1,gpos2);
		  
		  //if we have a doublet then search for the third hit
		  if(compatible) {
		    for ( iterRecHit3 = iterRecHit2+1; iterRecHit3 != theRecHitRangeIteratorEnd; ++iterRecHit3) {  
		      hit3 = &(*iterRecHit3);
		      if((unsigned int)hit3->geographicalId().subdetId()== PixelSubdetector::PixelBarrel) {
			const DetId& detId = hit3->geographicalId();
			PXBDetId pxbid3(detId.rawId()); 
			unsigned int layerNumberHit3 = pxbid3.layer();  
			if(layerNumberHit3 == 3 ){
			  //const GeomDet* geomDet( theGeometry->idToDet(detId) );
			  //GlobalPoint gpos3 = geomDet->surface().toGlobal(hit3->localPosition());
			  triplet = true;
			  TrackingRecHit* aTrackingRecHit1 = 
			    GenericTransientTrackingRecHit::build(geomDet,hit1)->hit()->clone();
			  TripletHits.push_back(aTrackingRecHit1);	
			  TrackingRecHit* aTrackingRecHit2 = 
			    GenericTransientTrackingRecHit::build(geomDet,hit2)->hit()->clone();
			  TripletHits.push_back(aTrackingRecHit2);	
			  TrackingRecHit* aTrackingRecHit3 = 
			    GenericTransientTrackingRecHit::build(geomDet,hit3)->hit()->clone();
			  TripletHits.push_back(aTrackingRecHit3);	
			  ++nTriplets;
			  if(triplet) break;
			}
		      } else if ( (unsigned int)hit3->geographicalId().subdetId()== PixelSubdetector::PixelEndcap  ) {
			//const DetId& detId = hit3->geographicalId();
			//const GeomDet* geomDet( theGeometry->idToDet(detId) );
			//GlobalPoint gpos3 = geomDet->surface().toGlobal(hit3->localPosition());
			triplet = true; 
			  TrackingRecHit* aTrackingRecHit1 = 
			    GenericTransientTrackingRecHit::build(geomDet,hit1)->hit()->clone();
			  TripletHits.push_back(aTrackingRecHit1);	
			  TrackingRecHit* aTrackingRecHit2 = 
			    GenericTransientTrackingRecHit::build(geomDet,hit2)->hit()->clone();
			  TripletHits.push_back(aTrackingRecHit2);	
			  TrackingRecHit* aTrackingRecHit3 = 
			    GenericTransientTrackingRecHit::build(geomDet,hit3)->hit()->clone();
			  TripletHits.push_back(aTrackingRecHit3);	
			  ++nTriplets;
			  if(triplet) break;
		      }
		    }
		  }
		  
		}
		//second hit in the first disk
	      } else if ((unsigned int)hit2->geographicalId().subdetId()== PixelSubdetector::PixelEndcap  ) {
		const DetId& detId = hit2->geographicalId();
		PXFDetId pxfid2(detId.rawId()); 
		unsigned int layerNumber2 = pxfid2.disk();
		if(layerNumber2 == 1) {
		  const GeomDet* geomDet( theGeometry->idToDet(detId) );
		  GlobalPoint gpos2 = geomDet->surface().toGlobal(hit2->localPosition());
		  compatible = compatibleWithVertex(gpos1,gpos2);
		  //search for the third hit in the second disk
		  if(compatible) {
		    for ( iterRecHit3 = iterRecHit2+1; iterRecHit3 != theRecHitRangeIteratorEnd; ++iterRecHit3) {  
		      hit3 = &(*iterRecHit3);
		      if((unsigned int)hit3->geographicalId().subdetId()== PixelSubdetector::PixelEndcap) {
			const DetId& detId = hit3->geographicalId();
			PXFDetId pxfid3(detId.rawId()); 
			unsigned int layerNumberHit3 = pxfid3.disk();  
			if(layerNumberHit3 == 2 ){ //second disk
			  triplet = true;
			  TrackingRecHit* aTrackingRecHit1 = 
			    GenericTransientTrackingRecHit::build(geomDet,hit1)->hit()->clone();
			  TripletHits.push_back(aTrackingRecHit1);	
			  TrackingRecHit* aTrackingRecHit2 = 
			    GenericTransientTrackingRecHit::build(geomDet,hit2)->hit()->clone();
			  TripletHits.push_back(aTrackingRecHit2);	
			  TrackingRecHit* aTrackingRecHit3 = 
			    GenericTransientTrackingRecHit::build(geomDet,hit3)->hit()->clone();
			  TripletHits.push_back(aTrackingRecHit3);	
			  ++nTriplets;
			  if(triplet) break;
			}
		      }
		    }
		  }
		}
	      }
	      
	    }
	  }
	}
      }//loop over rechits
      
      // fitting the triplet
      reco::Track* track = theFitter->run(es, TripletHits, region);
      
      // decide if track should be skipped according to filter 
      if ( ! (*theFilter)(track) ) { 
	delete track; 
	for ( unsigned i=0; i<TripletHits.size(); ++i ) delete TripletHits[i];
	continue; 
      }
      ++nFilterTracks;
      
      // add tracks 
      pixeltracks.push_back(TrackWithRecHits(track, TripletHits));
      ++nPixelTracks;
      
    }
  }
  
  int cc=0;
  int nTracks = pixeltracks.size();
  for (int i = 0; i < nTracks; i++)
  {
    //  reco::Track* track =  cleanedTracks.at(i).first;
    //const RecHits & hits = cleanedTracks.at(i).second;
    reco::Track* track   =  pixeltracks.at(i).first;
    const RecHits & hits = pixeltracks.at(i).second;
    
    for (unsigned int k = 0; k < hits.size(); k++)
      {
	TrackingRecHit *hit = (TrackingRecHit*)(hits.at(k));
	track->setHitPattern(*hit, k);
	recHits->push_back(hit);
      }
    tracks->push_back(*track);
    delete track;
    
  }
  
  edm::OrphanHandle <TrackingRecHitCollection> ohRH = e.put( recHits );
  
  for (int k = 0; k < nTracks; k++)
    {
      reco::TrackExtra* theTrackExtra = new reco::TrackExtra();
      
      //fill the TrackExtra with TrackingRecHitRef
      unsigned int nHits = tracks->at(k).numberOfValidHits();
      for(unsigned int i = 0; i < nHits; ++i) {
	theTrackExtra->add(TrackingRecHitRef(ohRH,cc));
	cc++;
      }
      
      trackExtras->push_back(*theTrackExtra);
      delete theTrackExtra;
    }
  
  edm::OrphanHandle<reco::TrackExtraCollection> ohTE = e.put(trackExtras);
  
  for (int k = 0; k < nTracks; k++)
    {
      const reco::TrackExtraRef theTrackExtraRef(ohTE,k);
      (tracks->at(k)).setExtra(theTrackExtraRef);
    }
  
  e.put(tracks);
  
  
  /*
  std::cout << " PixelTracksMaker: Total SimTracks           = " << nSimTracks <<  std::endl
	    << "                   Total SimTracksWithHits   = " << nTracksWithHits  <<  std::endl
	    << "                   Total SimTracksWithPT     = " << nTracksWithPT  <<  std::endl 
	    << "                   Total SimTracksWithD0Z0   = " << nTracksWithD0Z0  << std::endl 
	    << "                   Total Triplets            = " << nTriplets <<std::endl
	    << "                   Total Filtered Tracks     = " << nFilterTracks <<std::endl
	    << "                   Total Pixel Tracks        = " << nPixelTracks <<std::endl
	    << std::endl;
  */

  for ( unsigned ir=0; ir<regions.size(); ++ir ) delete regions[ir];

}


bool
PixelTracksMaker::compatibleWithVertex(GlobalPoint& gpos1, GlobalPoint& gpos2) {

  // The hits 1 and 2 positions, in HepLorentzVector's
  XYZTLorentzVector thePos1(gpos1.x(),gpos1.y(),gpos1.z(),0.);
  XYZTLorentzVector thePos2(gpos2.x(),gpos2.y(),gpos2.z(),0.);

  // Create new particles that pass through the second hit with pT = ptMin 
  // and charge = +/-1
  XYZTLorentzVector theMom2 = (thePos2-thePos1);

  theMom2 /= theMom2.Pt();
  theMom2 *= originpTMin;
  theMom2.SetE(sqrt(theMom2.Vect().Mag2()));

  // The corresponding RawParticles (to be propagated) for e- and e+
  ParticlePropagator myElecL(theMom2,thePos2,-1.);
  ParticlePropagator myPosiL(theMom2,thePos2,+1.);

  // Propagate to the closest approach point, with the constraint that 
  // the particles should pass through the  first hit
  myElecL.propagateToNominalVertex(thePos1);
  myPosiL.propagateToNominalVertex(thePos1);

  theMom2 *= 1000.0;//ptmax
  // The corresponding RawParticles (to be propagated) for e- and e+
  ParticlePropagator myElecH(theMom2,thePos2,-1.);
  ParticlePropagator myPosiH(theMom2,thePos2,+1.);

  // Propagate to the closest approach point, with the constraint that 
  // the particles should pass through the  first hit
  myElecH.propagateToNominalVertex(thePos1);
  myPosiH.propagateToNominalVertex(thePos1);

  // And check at least one of the particles statisfy the SeedGenerator
  // constraint (originRadius, originHalfLength)

  /*
  std::cout << " Neg Charge L R = " << myElecL.R() << "\t Z = " << fabs(myElecL.Z()) << std::endl;
  std::cout << " Pos Charge L R = " << myPosiL.R() << "\t Z = " << fabs(myPosiL.Z()) << std::endl;
  std::cout << " Neg Charge H R = " << myElecH.R() << "\t Z = " << fabs(myElecH.Z()) << std::endl;
  std::cout << " Pos Charge H R = " << myPosiH.R() << "\t Z = " << fabs(myPosiH.Z()) << std::endl;
  */

  if ( myElecL.R() < originRadius && 
       fabs(myElecL.Z()) < originHalfLength ) return true;
  if ( myPosiL.R() < originRadius && 
       fabs(myPosiL.Z()) < originHalfLength ) return true;
  if ( myElecH.R() < originRadius && 
       fabs(myElecH.Z()) < originHalfLength ) return true;
  if ( myPosiH.R() < originRadius && 
       fabs(myPosiH.Z()) < originHalfLength ) return true;

  return false;
}

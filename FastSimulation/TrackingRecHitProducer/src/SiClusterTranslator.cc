
/** SiClusterTranslator.cc
 * --------------------------------------------------------------
 * Description:  see SiClusterTranslator.h
 * Authors:  R. Ranieri (CERN)
 * History: Sep 27, 2006 -  initial version
 * --------------------------------------------------------------
 */

// SiTracker Gaussian Smearing
#include "SiClusterTranslator.h"

// Geometry
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"
#include "Geometry/CommonTopologies/interface/RectangularStripTopology.h"
#include "Geometry/CommonTopologies/interface/RadialStripTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"

//CPEs
#include "RecoLocalTracker/Records/interface/TkPixelCPERecord.h"
#include "RecoLocalTracker/Records/interface/TkStripCPERecord.h"
#include "FastPixelCPE.h"
#include "FastStripCPE.h"

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// Data Formats
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelDetId/interface/PixelChannelIdentifier.h"

//for the SimHit
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"

// STL
#include <memory>
#include <string>

SiClusterTranslator::SiClusterTranslator(edm::ParameterSet const& conf) :
  fastTrackerClusterCollectionTag_(conf.getParameter<edm::InputTag>("fastTrackerClusterCollectionTag"))
{
  produces<edmNew::DetSetVector<SiStripCluster> >();
  produces<edmNew::DetSetVector<SiPixelCluster> >();
  produces<edm::DetSetVector<StripDigiSimLink> >();
  produces<edm::DetSetVector<PixelDigiSimLink> >();
}

// Destructor
SiClusterTranslator::~SiClusterTranslator() {}  

void 
SiClusterTranslator::beginRun(edm::Run const&, const edm::EventSetup & es) {

  // Initialize the Tracker Geometry
  edm::ESHandle<TrackerGeometry> theGeometry;
  es.get<TrackerDigiGeometryRecord> ().get (theGeometry);
  geometry = &(*theGeometry);
}

void 
SiClusterTranslator::produce(edm::Event& e, const edm::EventSetup& es) 
{
  // Step A: Get Inputs (FastGSRecHit's)
  edm::Handle<FastTrackerClusterCollection> theFastClusters; 
  e.getByLabel(fastTrackerClusterCollectionTag_, theFastClusters);
  
  edm::ESHandle<TrackerGeometry> tkgeom;
  es.get<TrackerDigiGeometryRecord>().get( tkgeom ); 
  const TrackerGeometry &tracker(*tkgeom);
  
  edm::ESHandle<PixelClusterParameterEstimator> pixelCPE;
  es.get<TkPixelCPERecord>().get("FastPixelCPE",pixelCPE);
  auto pixelcpe = pixelCPE->clone();
  pixelcpe->clearParameters();
  
  edm::ESHandle<StripClusterParameterEstimator> stripCPE;
  es.get<TkStripCPERecord>().get("FastStripCPE", stripCPE); 
  auto stripcpe = stripCPE->clone();
    
  edm::Handle<CrossingFrame<PSimHit> > cf_simhit;
  std::vector<const CrossingFrame<PSimHit> *> cf_simhitvec;
  e.getByLabel("mix","famosSimHitsTrackerHits", cf_simhit);
  cf_simhitvec.push_back(cf_simhit.product());
  
  std::auto_ptr<MixCollection<PSimHit> > allTrackerHits(new MixCollection<PSimHit>(cf_simhitvec));
  int counter =0;
  
  //Clearing vector to hopefully make it run faster.
  theNewSimHitList.clear();
  thePixelDigiLinkVector.clear();
  theStripDigiLinkVector.clear();

  for(MixCollection<PSimHit>::iterator it = allTrackerHits->begin(); it!= allTrackerHits->end();it++){
    counter++;
    theNewSimHitList.push_back(std::make_pair((*it), counter));
  }

  // Step B: fill a temporary full Cluster collection from the fast Cluster collection
  FastTrackerClusterCollection::const_iterator aCluster = theFastClusters->begin();
  FastTrackerClusterCollection::const_iterator theLastHit = theFastClusters->end();
  std::map< DetId, std::vector<SiPixelCluster> > temporaryPixelClusters;
  std::map< DetId, std::vector<SiStripCluster> > temporaryStripClusters;
  
  //Clearing CPE maps from previous event.
  stripcpe->clearParameters();
  pixelcpe->clearParameters();
  
  int ClusterNum = 0;
  
  // loop on Fast GS Hits
  for ( ; aCluster != theLastHit; ++aCluster ) {
    ClusterNum++;
    
    //Finding SubDet Id of cluster: 1 & 2 = Pixel, 3 & 4 & 5 & 6 = Strip, >6 = ?
    DetId det = aCluster->id();
    unsigned int subdet   = det.subdetId();
    int sim_counter = 0;
    for (std::vector<std::pair<PSimHit,int> >::const_iterator 
	   simcount = theNewSimHitList.begin() ; simcount != theNewSimHitList.end(); simcount ++){
      if((aCluster->simtrackId() == (int)(*simcount).first.trackId())&&(det.rawId() == (*simcount).first.detUnitId())&&(aCluster->eeId() == (*simcount).first.eventId().rawId()))
	sim_counter = (*simcount).second;
    }
    if (sim_counter == 0)  throw cms::Exception("SiClusterTranslator") << "No Matching SimHit found.";
    
    //Separating into Pixels and Strips
    
    //Pixel
    if (subdet < 3) {
      //Here is the hard part. From the position of the FastSim Cluster I need to figure out the Pixel location for the Cluster.
      LocalPoint position = aCluster->localPosition();
      LocalError error = aCluster->localPositionError();
      //std::cout << "The pixel charge is " << aCluster->charge() << std::endl;
      int charge = (int)(aCluster->charge() + 0.5);
      //std::cout << "The pixel charge after integer conversion is " << charge << std::endl;

      //std::vector<int> digi_vec;
      //while (charge > 255) { 
      //digi_vec.push_back(charge);
      //charge -= 256;
      //}
      //digi_vec.push_back(charge);
      
      const GeomDetUnit *  geoDet = tracker.idToDetUnit(det);
      const PixelGeomDetUnit * pixelDet=(const PixelGeomDetUnit*)(geoDet);
      const PixelTopology& topol=(const PixelTopology&)pixelDet->topology();
      
      //Out of pixel is float, but input of pixel is int. Hopeful it works...
      std::pair<float,float> pixelPos_out = topol.pixel(position);
      SiPixelCluster::PixelPos pixelPos((int)pixelPos_out.first, (int)pixelPos_out.second);
      
      //Filling Pixel CPE with information.
      std::pair<int,int> row_col((int)pixelPos_out.first,(int)pixelPos_out.second);
      pixelcpe->enterLocalParameters((unsigned int) det.rawId() , row_col, std::make_pair(position,error));
      
      unsigned int ch = PixelChannelIdentifier::pixelToChannel((int)pixelPos_out.first, (int)pixelPos_out.second);
      
      //Creating a new pixel cluster.
      SiPixelCluster temporaryPixelCluster(pixelPos, charge);
      temporaryPixelClusters[det].push_back(temporaryPixelCluster);
      
      //Making a PixelDigiSimLink.
      edm::DetSet<PixelDigiSimLink> pixelDetSet;
      pixelDetSet.id = det.rawId();
      pixelDetSet.data.push_back(PixelDigiSimLink(ch,
						  aCluster->simtrackId(),
						  EncodedEventId(aCluster->eeId()),
						  1.0));
      thePixelDigiLinkVector.push_back(pixelDetSet);
    } 
    
    //Strips
    else if ((subdet > 2) && (subdet < 7)) {
      //Getting pos/err info from the cluster
      LocalPoint position = aCluster->localPosition();
      LocalError error = aCluster->localPositionError();
      
      //Will have to make charge into ADC eventually...
      uint16_t charge = (uint16_t)(aCluster->charge() + 0.5);
      
      //std::cout << "The charge is " << charge << std::endl;
      
      uint16_t strip_num = 0;
      std::vector<uint16_t> digi_vec;
      while (charge > 255) { 
	digi_vec.push_back(255);
	charge -= 255;
      }
      if (charge > 0) digi_vec.push_back(charge);
      //std::cout << "The digi_vec size is " << digi_vec.size() << std::endl;
      //int totcharge = 0;
      //for(int i = 0; i < digi_vec.size(); ++i) {
      //totcharge += digi_vec[i];
      //} 
      const GeomDetUnit *  geoDet = tracker.idToDetUnit(det);
      const StripGeomDetUnit * stripDet = (const StripGeomDetUnit*)(geoDet);
      
      //3 = TIB, 4 = TID, 5 = TOB, 6 = TEC
      if((subdet == 3) || (subdet == 5)) {
	const RectangularStripTopology& topol=(const RectangularStripTopology&)stripDet->type().topology();
	strip_num = (uint16_t)topol.strip(position);
      } else if ((subdet == 4) || (subdet == 6)) {
	const RadialStripTopology& topol=(const RadialStripTopology&)stripDet->type().topology();
	strip_num = (uint16_t)topol.strip(position);
      }
      
      //Filling Strip CPE with info.
      stripcpe->enterLocalParameters(det.rawId(), strip_num, std::make_pair(position,error));
      
      //Creating a new strip cluster
      SiStripCluster temporaryStripCluster(strip_num, digi_vec.begin(), digi_vec.end());
      temporaryStripClusters[det].push_back(temporaryStripCluster);
      
      //Making a StripDigiSimLink
      edm::DetSet<StripDigiSimLink> stripDetSet;
      stripDetSet.id = det.rawId();
      stripDetSet.data.push_back(StripDigiSimLink(strip_num,
						  aCluster->simtrackId(),
						  sim_counter,
						  EncodedEventId(aCluster->eeId()),
						  1.0));
      theStripDigiLinkVector.push_back(stripDetSet);
    } 
    
    //?????
    else {
      throw cms::Exception("SiClusterTranslator") <<
	"Trying to build a cluster that is not in the SiStripTracker or Pixels.\n";
    }
    
  }//Cluster loop

  // Step C: from the temporary Cluster collections, create the real ones.
  
  //Pixels
  std::auto_ptr<edmNew::DetSetVector<SiPixelCluster> >
    siPixelClusterCollection(new edmNew::DetSetVector<SiPixelCluster>);
  loadPixelClusters(temporaryPixelClusters, *siPixelClusterCollection);
  std::auto_ptr<edm::DetSetVector<StripDigiSimLink> > stripoutputlink(new edm::DetSetVector<StripDigiSimLink>(theStripDigiLinkVector) );
  
  
  //Strips
  std::auto_ptr<edmNew::DetSetVector<SiStripCluster> > 
    siStripClusterCollection(new edmNew::DetSetVector<SiStripCluster>);
  loadStripClusters(temporaryStripClusters, *siStripClusterCollection);
  std::auto_ptr<edm::DetSetVector<PixelDigiSimLink> > pixeloutputlink(new edm::DetSetVector<PixelDigiSimLink>(thePixelDigiLinkVector) );
  
  // Step D: write output to file
  e.put(siPixelClusterCollection);
  e.put(siStripClusterCollection);
  e.put(stripoutputlink);
  e.put(pixeloutputlink);
}

void 
SiClusterTranslator::loadStripClusters(
				       std::map<DetId,std::vector<SiStripCluster> >& theClusters,
				       edmNew::DetSetVector<SiStripCluster>& theClusterCollection) const
{
  std::map<DetId,std::vector<SiStripCluster> >::const_iterator 
    it = theClusters.begin();
  std::map<DetId,std::vector<SiStripCluster> >::const_iterator 
    lastDet = theClusters.end();
  for( ; it != lastDet ; ++it ) { 
    edmNew::DetSetVector<SiStripCluster>::FastFiller cluster_col(theClusterCollection, it->first);
    
    std::vector<SiStripCluster>::const_iterator clust_it = it->second.begin();
    std::vector<SiStripCluster>::const_iterator clust_end = it->second.end();
    for( ; clust_it != clust_end ; ++clust_it)  {
      cluster_col.push_back(*clust_it);
    }
  }
}

void 
SiClusterTranslator::loadPixelClusters(
				       std::map<DetId,std::vector<SiPixelCluster> >& theClusters,
				       edmNew::DetSetVector<SiPixelCluster>& theClusterCollection) const
{
  
  std::map<DetId,std::vector<SiPixelCluster> >::const_iterator 
    it = theClusters.begin();
  std::map<DetId,std::vector<SiPixelCluster> >::const_iterator 
    lastCluster = theClusters.end();
  for( ; it != lastCluster ; ++it ) { 
    edmNew::DetSetVector<SiPixelCluster>::FastFiller spc(theClusterCollection, it->first);
    
    std::vector<SiPixelCluster>::const_iterator clust_it = it->second.begin();
    std::vector<SiPixelCluster>::const_iterator clust_end = it->second.end();
    for( ; clust_it != clust_end ; ++clust_it)  {
      spc.push_back(*clust_it);
    }
  }
}

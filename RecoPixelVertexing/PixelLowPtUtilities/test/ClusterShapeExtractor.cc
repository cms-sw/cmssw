// VI January 2012: needs to be migrated to use cluster directly

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelClusterShapeCache.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShapeHitFilter.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"

#include "TROOT.h"
#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"

#include <utility>
#include <vector>
#include <fstream>
#include <memory>

using namespace std;

// pixel
#define exMax 10
#define eyMax 15

// strip
#define ewMax 40

/*****************************************************************************/
class ClusterShapeExtractor : public edm::EDAnalyzer
{
 public:
   explicit ClusterShapeExtractor(const edm::ParameterSet& pset);
   ~ClusterShapeExtractor();
   virtual void beginRun(const edm::Run & run, const edm::EventSetup& es) override;
   virtual void analyze (const edm::Event& ev, const edm::EventSetup& es) override;
   virtual void endJob() override;

 private:
   bool isSuitable(const PSimHit & simHit);

   // Sim
   void processSim(const SiPixelRecHit &   recHit,
                   const PSimHit & simHit, const SiPixelClusterShapeCache& clusterShapeCache, vector<TH2F *> & hspc);
   void processSim(const SiStripRecHit2D & recHit,
                   const PSimHit & simHit, vector<TH1F *> & hssc);

   // Rec
   void processRec(const SiPixelRecHit &   recHit,
                   LocalVector ldir, const SiPixelClusterShapeCache& clusterShapeCache, vector<TH2F *> & hrpc);
   void processRec(const SiStripRecHit2D & recHit,
                   LocalVector ldir, vector<TH1F *> & hrsc);

   bool checkSimHits
    (const TrackingRecHit & recHit, PSimHit & simHit,
     pair<unsigned int, float> & key);

   void processPixelRecHits
     (const SiPixelRecHitCollection::DataContainer * recHits, const SiPixelClusterShapeCache& clusterShapeCache);
   void processStripRecHits
     (const SiStripRecHit2DCollection::DataContainer * recHits);
   void processMatchedRecHits
     (const SiStripMatchedRecHit2DCollection::DataContainer * recHits);

   void analyzeSimHits  (const edm::Event& ev, const edm::EventSetup& es);
   void analyzeRecTracks(const edm::Event& ev, const edm::EventSetup& es);

   TFile * file;

   string trackProducer;
   bool hasSimHits;
   bool hasRecTracks;

   edm::EDGetTokenT<SiPixelClusterShapeCache> theClusterShapeCacheToken;

   const TrackerGeometry * theTracker;
   std::unique_ptr<TrackerHitAssociator> theHitAssociator;
   TrackerHitAssociator::Config trackerHitAssociatorConfig_;
   const ClusterShapeHitFilter * theClusterShape;

   vector<TH2F *> hspc; // simulated pixel cluster
   vector<TH1F *> hssc; // simulated strip cluster

   vector<TH2F *> hrpc; // reconstructed pixel cluster
   vector<TH1F *> hrsc; // reconstructed strip cluster
};

/*****************************************************************************/
void ClusterShapeExtractor::beginRun(const edm::Run & run, const edm::EventSetup& es)
{
  // Get tracker geometry
  edm::ESHandle<TrackerGeometry>          tracker;
  es.get<TrackerDigiGeometryRecord>().get(tracker);
  theTracker =                            tracker.product();

  // 
  //  theClusterShape = new ClusterShapeHitFilter(es);
  edm::ESHandle<ClusterShapeHitFilter> shape;
  es.get<CkfComponentsRecord>().get("ClusterShapeHitFilter",shape);
  theClusterShape = shape.product();

  // Declare histograms
  char histName[256];

  // pixel
  for(int subdet = 0; subdet <= 1; subdet++)
  {
    for(int ex = 0; ex <= exMax; ex++)
    for(int ey = 0; ey <= eyMax; ey++)
    {
      sprintf(histName,"hspc_%d_%d_%d",subdet, ex,ey);
      hspc.push_back(new TH2F(histName,histName, 
                         10 * 2 * (exMax+2), -(exMax+2),(exMax+2),
                         10 * 2 * (eyMax+2), -(eyMax+2),(eyMax+2)));

      sprintf(histName,"hrpc_%d_%d_%d",subdet, ex,ey);
      hrpc.push_back(new TH2F(histName,histName, 
                         10 * 2 * (exMax+2), -(exMax+2),(exMax+2),
                         10 * 2 * (eyMax+2), -(eyMax+2),(eyMax+2)));
    }
  }

  // strip
  for(int ew = 0; ew <= ewMax; ew++)
  {
    sprintf(histName,"hssc_%d", ew);
    hssc.push_back(new TH1F(histName,histName,
                       10 * 2 * (ewMax*2), -(ewMax*2),(ewMax*2)));

    sprintf(histName,"hrsc_%d", ew);
    hrsc.push_back(new TH1F(histName,histName,
                       10 * 2 * (ewMax*2), -(ewMax*2),(ewMax*2)));
  }
}

/*****************************************************************************/
ClusterShapeExtractor::ClusterShapeExtractor
  (const edm::ParameterSet& pset) : theClusterShapeCacheToken(consumes<SiPixelClusterShapeCache>(pset.getParameter<edm::InputTag>("clusterShapeCacheSrc"))),
  trackerHitAssociatorConfig_(pset, consumesCollector())
{
  trackProducer = pset.getParameter<string>("trackProducer"); 
  hasSimHits    = pset.getParameter<bool>("hasSimHits"); 
  hasRecTracks  = pset.getParameter<bool>("hasRecTracks"); 

  file = new TFile("clusterShape.root","RECREATE");
  file->cd();
}

/*****************************************************************************/
void ClusterShapeExtractor::endJob()
{
  typedef vector<TH2F *>::const_iterator H2I;
  typedef vector<TH1F *>::const_iterator H1I;

  file->cd();

  // simulated
  for(H2I h = hspc.begin(); h!= hspc.end(); h++) (*h)->Write();
  for(H1I h = hssc.begin(); h!= hssc.end(); h++) (*h)->Write();

  // reconstructed
  for(H2I h = hrpc.begin(); h!= hrpc.end(); h++) (*h)->Write();
  for(H1I h = hrsc.begin(); h!= hrsc.end(); h++) (*h)->Write();

  file->Close();
}

/*****************************************************************************/
ClusterShapeExtractor::~ClusterShapeExtractor()
{
//  delete theClusterShape;
}

/*****************************************************************************/
bool ClusterShapeExtractor::isSuitable(const PSimHit & simHit)
{
  // Outgoing?
  DetId id = DetId(simHit.detUnitId());
  const GeomDetUnit *gdu = theTracker->idToDetUnit(id);
  if (gdu == 0) throw cms::Exception("MissingData") << "Missing DetUnit for detid " << id.rawId() << "\n" << std::endl;
  GlobalVector gvec = theTracker->idToDetUnit(id)->position() -
                      GlobalPoint(0,0,0);
  LocalVector  lvec = theTracker->idToDetUnit(id)->toLocal(gvec);
  LocalVector  ldir = simHit.exitPoint() - simHit.entryPoint();

  bool isOutgoing = (lvec.z()*ldir.z() > 0); 

  // From a relevant process? primary or decay
  bool isRelevant = (simHit.processType() == 2 ||
                     simHit.processType() == 4);

  // Fast enough? pt > 50 MeV/c
  bool isFast = (simHit.momentumAtEntry().perp() > 0.050);

  return (isOutgoing && isRelevant && isFast);
}

/*****************************************************************************/
void ClusterShapeExtractor::processRec(const SiStripRecHit2D & recHit,
     LocalVector ldir, vector<TH1F *> & histo)
{
  int meas;
  float pred;
 
  if(theClusterShape->getSizes(recHit,LocalPoint(0,0,0), ldir, meas,pred))
    if(meas <= ewMax)
      histo[meas]->Fill(pred);
}

/*****************************************************************************/
void ClusterShapeExtractor::processRec(const SiPixelRecHit & recHit,
    LocalVector ldir, const SiPixelClusterShapeCache& clusterShapeCache, vector<TH2F *> & histo)
{
  int part;
  ClusterData::ArrayType meas;
  pair<float,float> pred;
 
  if(theClusterShape->getSizes(recHit,ldir,clusterShapeCache, part,meas,pred))
   if(meas.size() == 1)
    if(meas.front().first  <= exMax && 
       meas.front().second <= eyMax)
    {
      int i = (part * (exMax + 1) +
               meas.front().first) * (eyMax + 1) +
               meas.front().second;
      histo[i]->Fill(pred.first, pred.second);
    }
}

/*****************************************************************************/
void ClusterShapeExtractor::processSim(const SiPixelRecHit & recHit,
     const PSimHit & simHit, const SiPixelClusterShapeCache& clusterShapeCache, vector<TH2F *> & histo)
{
  LocalVector ldir = simHit.exitPoint() - simHit.entryPoint();
  processRec(recHit, ldir, clusterShapeCache, histo);
}

/*****************************************************************************/
void ClusterShapeExtractor::processSim(const SiStripRecHit2D & recHit,
     const PSimHit & simHit, vector<TH1F *> & histo)
{
  LocalVector ldir = simHit.exitPoint() - simHit.entryPoint();
  processRec(recHit, ldir, histo);
}

/*****************************************************************************/
bool ClusterShapeExtractor::checkSimHits
  (const TrackingRecHit & recHit, PSimHit & simHit,
   pair<unsigned int, float> & key)
{
  vector<PSimHit> simHits = theHitAssociator->associateHit(recHit);

  if(simHits.size() == 1)
  {
    simHit = simHits[0];

    if(isSuitable(simHit))
    {
      key = pair<unsigned int, float>(simHit.trackId(),
                                      simHit.timeOfFlight());
      return true;
    }
  } 

  return false;
}

/*****************************************************************************/
void ClusterShapeExtractor::processPixelRecHits
  (const SiPixelRecHitCollection::DataContainer * recHits, const SiPixelClusterShapeCache& clusterShapeCache)
{
  map<pair<unsigned int, float>, const SiPixelRecHit *> simHitMap;

  PSimHit simHit;
  pair<unsigned int, float> key;

  for(  SiPixelRecHitCollection::DataContainer::const_iterator
        recHit = recHits->begin(); recHit!= recHits->end(); recHit++)
  if(checkSimHits(*recHit, simHit, key))
  {
    // Fill map
    if(simHitMap.count(key) == 0)
       simHitMap[key] = &(*recHit);
    else
      if(        recHit->cluster()->size() >
         simHitMap[key]->cluster()->size())
         simHitMap[key] = &(*recHit);
  }

  for(  SiPixelRecHitCollection::DataContainer::const_iterator
        recHit = recHits->begin(); recHit!= recHits->end(); recHit++)
  if(checkSimHits(*recHit, simHit, key))
  {
    // Check whether the present rechit is the largest
    if(&(*recHit) == simHitMap[key])
      processSim(*recHit, simHit, clusterShapeCache, hspc);
  }
}

/*****************************************************************************/
void ClusterShapeExtractor::processStripRecHits
  (const SiStripRecHit2DCollection::DataContainer * recHits)
{
  map<pair<unsigned int, float>, const SiStripRecHit2D *> simHitMap;

  PSimHit simHit;
  pair<unsigned int, float> key;

  for(  SiStripRecHit2DCollection::DataContainer::const_iterator
        recHit = recHits->begin(); recHit!= recHits->end(); recHit++)
  if(checkSimHits(*recHit, simHit, key))
  {
    // Fill map
    if(simHitMap.count(key) == 0)
       simHitMap[key] = &(*recHit);
    else
      if(        recHit->cluster()->amplitudes().size() >
         simHitMap[key]->cluster()->amplitudes().size())
         simHitMap[key] = &(*recHit);
  }

  for(  SiStripRecHit2DCollection::DataContainer::const_iterator
        recHit = recHits->begin(); recHit!= recHits->end(); recHit++)
  if(checkSimHits(*recHit, simHit, key))
  {
    // Check whether the present rechit is the largest
    if(&(*recHit) == simHitMap[key])
      processSim(*recHit, simHit, hssc);
  }
}

/*****************************************************************************/
void ClusterShapeExtractor::processMatchedRecHits
  (const SiStripMatchedRecHit2DCollection::DataContainer * recHits)
{
  map<pair<unsigned int, float>, const SiStripRecHit2D *> simHitMap;
  // VI very very quick fix
  std::vector<SiStripRecHit2D> cache;
  cache.reserve(2*recHits->size());
  PSimHit simHit;
  pair<unsigned int, float> key;

  const SiStripRecHit2D * recHit;

  for(  SiStripMatchedRecHit2DCollection::DataContainer::const_iterator
        matched = recHits->begin(); matched!= recHits->end(); matched++)
  {
    cache.push_back(matched->monoHit());
    recHit = &cache.back();
    if(checkSimHits(*recHit, simHit, key))
    {
      // Fill map
      if(simHitMap.count(key) == 0)
         simHitMap[key] = &(*recHit);
      else
        if(        recHit->cluster()->amplitudes().size() >
           simHitMap[key]->cluster()->amplitudes().size()) 
           simHitMap[key] = &(*recHit);
    }
    cache.push_back(matched->stereoHit());
    recHit = &cache.back();
    if(checkSimHits(*recHit, simHit, key))
    {
      // Fill map
      if(simHitMap.count(key) == 0)
         simHitMap[key] = &(*recHit);
      else
        if(        recHit->cluster()->amplitudes().size() >
           simHitMap[key]->cluster()->amplitudes().size())
           simHitMap[key] = &(*recHit);
    }

  }
  
  for(  SiStripMatchedRecHit2DCollection::DataContainer::const_iterator 
        matched = recHits->begin(); matched!= recHits->end(); matched++)
  {
    auto recHit = matched->monoHit();
    if(checkSimHits(recHit, simHit, key))
    {
      // Check whether the present rechit is the largest
      if(recHit.omniCluster() == simHitMap[key]->omniCluster()) 
        processSim(recHit, simHit, hssc);
    }

    recHit = matched->stereoHit();
    if(checkSimHits(recHit, simHit, key))
    {
      // Check whether the present rechit is the largest
     if(recHit.omniCluster() == simHitMap[key]->omniCluster())
        processSim(recHit, simHit, hssc);
    }
  }

}

/*****************************************************************************/
void ClusterShapeExtractor::analyzeSimHits
  (const edm::Event& ev, const edm::EventSetup& es)
{
  // Get associator
  theHitAssociator.reset(new TrackerHitAssociator(ev,trackerHitAssociatorConfig_));

  // Pixel hits
  {
    edm::Handle<SiPixelRecHitCollection> coll;
    ev.getByLabel("siPixelRecHits", coll);
  
    edm::Handle<SiPixelClusterShapeCache> clusterShapeCache;
    ev.getByToken(theClusterShapeCacheToken, clusterShapeCache);

    const SiPixelRecHitCollection::DataContainer * recHits =
          & coll.product()->data();
    processPixelRecHits(recHits, *clusterShapeCache);
  }

  // Strip hits
  { // rphi and stereo
    vector<edm::Handle<SiStripRecHit2DCollection> > colls;
    ev.getManyByType(colls);
  
    for(vector<edm::Handle<SiStripRecHit2DCollection> >::const_iterator
          coll = colls.begin(); coll!= colls.end(); coll++)
    {
      const SiStripRecHit2DCollection::DataContainer * recHits =
            & (*coll).product()->data();
      processStripRecHits(recHits);
    }
  }

  // Matched strip hits
  { // matched
  vector<edm::Handle<SiStripMatchedRecHit2DCollection> > colls;
  ev.getManyByType(colls);

  for(vector<edm::Handle<SiStripMatchedRecHit2DCollection> >::const_iterator
        coll = colls.begin(); coll!= colls.end(); coll++)
  {
    const SiStripMatchedRecHit2DCollection::DataContainer * recHits =
          & (*coll).product()->data();
    processMatchedRecHits(recHits);
  }
  }
}

/*****************************************************************************/
void ClusterShapeExtractor::analyzeRecTracks
  (const edm::Event& ev, const edm::EventSetup& es)
{
  // Get trajectory collection
  edm::Handle<vector<Trajectory> > trajeHandle;
  ev.getByLabel(trackProducer,     trajeHandle);
  const vector<Trajectory> & trajeCollection = *(trajeHandle.product());

  edm::Handle<SiPixelClusterShapeCache> clusterShapeCache;
  ev.getByToken(theClusterShapeCacheToken, clusterShapeCache);

  // Take all trajectories
  for(vector<Trajectory>::const_iterator trajectory = trajeCollection.begin();
                                         trajectory!= trajeCollection.end();
                                         trajectory++)
  for(vector<TrajectoryMeasurement>::const_iterator
      meas = trajectory->measurements().begin();
      meas!= trajectory->measurements().end(); meas++)
  {
    const TrackingRecHit* recHit = meas->recHit()->hit();
    DetId id = recHit->geographicalId();

    if(recHit->isValid())
    {
      LocalVector ldir = meas->updatedState().localDirection();

      if(GeomDetEnumerators::isTrackerPixel(theTracker->geomDetSubDetector(id.subdetId())))
      {
        // Pixel
        const SiPixelRecHit* pixelRecHit =
          dynamic_cast<const SiPixelRecHit *>(recHit);

        if(pixelRecHit != 0)
          processRec(*pixelRecHit, ldir, *clusterShapeCache, hrpc);
      }
      else if(GeomDetEnumerators::isTrackerStrip(theTracker->geomDetSubDetector(id.subdetId())))
      {
        // Strip
        const SiStripMatchedRecHit2D* stripMatchedRecHit =
          dynamic_cast<const SiStripMatchedRecHit2D *>(recHit);
        const ProjectedSiStripRecHit2D* stripProjectedRecHit =
          dynamic_cast<const ProjectedSiStripRecHit2D *>(recHit);
        const SiStripRecHit2D* stripRecHit =
          dynamic_cast<const SiStripRecHit2D *>(recHit);

        if(stripMatchedRecHit != 0)
        {
          processRec(stripMatchedRecHit->monoHit(), ldir, hrsc);
          processRec(stripMatchedRecHit->stereoHit(), ldir, hrsc);
        }

        if(stripProjectedRecHit != 0)
          processRec(stripProjectedRecHit->originalHit(), ldir, hrsc);

        if(stripRecHit != 0)
          processRec(*stripRecHit, ldir, hrsc);
      }
    }
  }
}

/*****************************************************************************/
void ClusterShapeExtractor::analyze
  (const edm::Event& ev, const edm::EventSetup& es)
{
  if(hasSimHits)
  {
    LogTrace("MinBiasTracking")
      << " [ClusterShape] analyze simHits, recHits";
    analyzeSimHits(ev, es);
  } 

  if(hasRecTracks)
  {
    LogTrace("MinBiasTracking") 
      << " [ClusterShape] analyze recHits on recTracks";
    analyzeRecTracks(ev,es);
  } 
}

DEFINE_FWK_MODULE(ClusterShapeExtractor);


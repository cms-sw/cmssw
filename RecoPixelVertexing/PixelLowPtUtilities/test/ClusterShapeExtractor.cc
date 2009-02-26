#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"

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
   virtual void beginJob(const edm::EventSetup& es);
   virtual void analyze (const edm::Event& ev, const edm::EventSetup& es);
   virtual void endJob();

 private:
   bool isSuitable(const PSimHit* simHit);

   // Sim
   void processSim(const SiPixelRecHit &   recHit,
                   vector<PSimHit> simHits, vector<TH2F *> & hspc);
   void processSim(const SiStripRecHit2D & recHit,
                   vector<PSimHit> simHits, vector<TH1F *> & hssc);

   // Rec
   void processRec(const SiPixelRecHit &   recHit,
                   LocalVector ldir, vector<TH2F *> & hrpc);
   void processRec(const SiStripRecHit2D & recHit,
                   LocalVector ldir, vector<TH1F *> & hrsc);

   void analyzeSimHits  (const edm::Event& ev, const edm::EventSetup& es);
   void analyzeRecTracks(const edm::Event& ev, const edm::EventSetup& es);

   TFile * file;

   edm::ParameterSet theConfig;
   string trackProducer;
   bool hasSimHits;
   bool hasRecTracks;

   const TrackerGeometry * theTracker;
   TrackerHitAssociator  * theHitAssociator;
   ClusterShapeHitFilter * theClusterShape;

   vector<TH2F *> hspc; // simulated pixel cluster
   vector<TH1F *> hssc; // simulated strip cluster

   vector<TH2F *> hrpc; // reconstructed pixel cluster
   vector<TH1F *> hrsc; // reconstructed strip cluster
};

/*****************************************************************************/
void ClusterShapeExtractor::beginJob(const edm::EventSetup& es)
{
  // Get tracker geometry
  edm::ESHandle<TrackerGeometry>          tracker;
  es.get<TrackerDigiGeometryRecord>().get(tracker);
  theTracker =                            tracker.product();

  // 
  theClusterShape = new ClusterShapeHitFilter(es);

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
  (const edm::ParameterSet& pset) : theConfig(pset)
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
  delete theClusterShape;
}

/*****************************************************************************/
bool ClusterShapeExtractor::isSuitable(const PSimHit* simHit)
{
  // Is it outgoing ?
  DetId id = DetId(simHit->detUnitId());

  GlobalVector gvec = theTracker->idToDetUnit(id)->position() -
                      GlobalPoint(0,0,0);
  LocalVector  lvec = theTracker->idToDetUnit(id)->toLocal(gvec);
  LocalVector  ldir = simHit->exitPoint() - simHit->entryPoint();

  bool isOutgoing = (lvec.z()*ldir.z() > 0); 

  // Is it from a relevant process and particle?
  bool isRelevant;
  if(simHit->processType() == 2 || // Primary
    (simHit->processType() == 3 && // Hadronic
     simHit->particleType() != -100 &&
     simHit->particleType() != -101 &&
     simHit->particleType() != -102) ||
     simHit->processType() == 4) // Decay
    isRelevant = true;
  else
    isRelevant = false;

  // Is fast enough?, pt > 50 MeV/c
  bool isFast = (simHit->momentumAtEntry().perp() > 0.050);

  return (isOutgoing && isRelevant && isFast);
}

/*****************************************************************************/
void ClusterShapeExtractor::processRec(const SiStripRecHit2D & recHit,
     LocalVector ldir, vector<TH1F *> & histo)
{
  int meas;
  float pred;
 
  if(theClusterShape->getSizes(recHit,ldir, meas,pred))
    if(meas <= ewMax)
      histo[meas]->Fill(pred);
}

/*****************************************************************************/
void ClusterShapeExtractor::processRec(const SiPixelRecHit & recHit,
     LocalVector ldir, vector<TH2F *> & histo)
{
  int part;
  pair<int,int> meas;
  pair<float,float> pred;
 
  if(theClusterShape->getSizes(recHit,ldir, part,meas,pred))
    if(meas.first  <= exMax && 
       meas.second <= eyMax)
    {
      int i = (part * (exMax + 1) + meas.first) * (eyMax + 1) + meas.second;
      histo[i]->Fill(meas.first, meas.second);
    }
}

/*****************************************************************************/
void ClusterShapeExtractor::processSim(const SiPixelRecHit & recHit,
     vector<PSimHit> simHits, vector<TH2F *> & histo)
{
  if(simHits.size() == 1)
  {
    const PSimHit* simHit = &(simHits[0]);

    if(isSuitable(simHit))
    {
      LocalVector ldir = simHit->exitPoint() - simHit->entryPoint();
      processRec(recHit, ldir, histo);
    }
  }
}

/*****************************************************************************/
void ClusterShapeExtractor::processSim(const SiStripRecHit2D & recHit,
  vector<PSimHit> simHits, vector<TH1F *> & histo)
{
  if(simHits.size() == 1)
  {
    const PSimHit* simHit = &(simHits[0]);

    if(isSuitable(simHit))
    {
      LocalVector ldir = simHit->exitPoint() - simHit->entryPoint();

      processRec(recHit, ldir, histo);
    }
  }
}

/*****************************************************************************/
void ClusterShapeExtractor::analyzeSimHits
  (const edm::Event& ev, const edm::EventSetup& es)
{
  // Get associator
  theHitAssociator = new
    TrackerHitAssociator::TrackerHitAssociator(ev,theConfig);

  // Pixel hits
  {
  edm::Handle<SiPixelRecHitCollection> coll;
  ev.getByLabel("siPixelRecHits", coll);

  const SiPixelRecHitCollection::DataContainer * recHits =
        & coll.product()->data();

  for(  SiPixelRecHitCollection::DataContainer::const_iterator
        recHit = recHits->begin(); recHit!= recHits->end(); recHit++)
  { 
    theHitAssociator->associateHit(*recHit);
    processSim(*recHit, theHitAssociator->associateHit(*recHit), hspc);
  }
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

    for(  SiStripRecHit2DCollection::DataContainer::const_iterator
          recHit = recHits->begin(); recHit!= recHits->end(); recHit++)
      processSim(*recHit, theHitAssociator->associateHit(*recHit), hssc);
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

    for(  SiStripMatchedRecHit2DCollection::DataContainer::const_iterator
          recHit = recHits->begin(); recHit!= recHits->end(); recHit++)
    {
      processSim(*(recHit->monoHit()),
                 theHitAssociator->associateHit(*(recHit->monoHit())), hssc);
      processSim(*(recHit->stereoHit()),
                 theHitAssociator->associateHit(*(recHit->stereoHit())), hssc);
    }
  }
  }

  delete theHitAssociator;
}

/*****************************************************************************/
void ClusterShapeExtractor::analyzeRecTracks
  (const edm::Event& ev, const edm::EventSetup& es)
{
  // Get trajectory collection
  edm::Handle<vector<Trajectory> > trajeHandle;
  ev.getByLabel(trackProducer,     trajeHandle);
  const vector<Trajectory> & trajeCollection = *(trajeHandle.product());

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

      if(theTracker->idToDet(id)->subDetector() ==
           GeomDetEnumerators::PixelBarrel ||
         theTracker->idToDet(id)->subDetector() ==
           GeomDetEnumerators::PixelEndcap)
      {
        // Pixel
        const SiPixelRecHit* pixelRecHit =
          dynamic_cast<const SiPixelRecHit *>(recHit);

        if(pixelRecHit != 0)
          processRec(*pixelRecHit, ldir, hrpc);
      }
      else
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
          processRec(*(stripMatchedRecHit->monoHit())  , ldir, hrsc);
          processRec(*(stripMatchedRecHit->stereoHit()), ldir, hrsc);
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
      << "[ClusterShape] analyze simHits, recHits";
    analyzeSimHits(ev, es);
  } 

  if(hasRecTracks)
  {
    LogTrace("MinBiasTracking") 
      << "[ClusterShape] analyze recHits on recTracks";
    analyzeRecTracks(ev,es);
  } 
}

DEFINE_FWK_MODULE(ClusterShapeExtractor);


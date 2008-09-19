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

#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterData.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShape.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"

#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

#include "Geometry/TrackerTopology/interface/RectangularPixelTopology.h"
#include "Geometry/CommonTopologies/interface/RectangularStripTopology.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "CondFormats/DataRecord/interface/SiStripLorentzAngleRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"

#include "TROOT.h"
#include "TFile.h"
#include "TH2F.h"
#include "TNtuple.h"

#include <utility>
#include <vector>
#include <fstream>
using namespace std;

// pixel
#define exMax  5
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
   void analyzeSimHits (const edm::Event& ev, const edm::EventSetup& es);
   void analyzeRecHits (const edm::Event& ev, const edm::EventSetup& es);
   virtual void analyze(const edm::Event& ev, const edm::EventSetup& es);
   virtual void endJob() { }

 private:
   bool isSuitable(const PSimHit* simHit);
   bool getOrientation(const SiStripRecHit2D& recHit, float& tangent, int& nstrips, bool& isBarrel);

   // Sim
   void process(const SiPixelRecHit*   recHit, vector<PSimHit> simHits);
   void process(const SiStripRecHit2D* recHit, vector<PSimHit> simHits);

   // Rec
   void process(LocalVector ldir, const SiPixelRecHit*   recHit);
   void process(LocalVector ldir, const SiStripRecHit2D* recHit);

   TFile * file;

   const TrackerGeometry* theTracker;
   const MagneticField* theMagneticField;
   const SiStripLorentzAngle * theLorentzAngle;

   const SiPixelRecHitCollection* recHits_;

   edm::ParameterSet theConfig;

   string trackProducer;
   bool hasSimInfo;

   int nhits;
   TrackerHitAssociator * theHitAssociator;

   std::vector<TH2F *> hspc; // simulated pixel cluster
   std::vector<TH1F *> hssc; // simulated strip cluster

   std::vector<TH2F *> hrpc; // reconstructed pixel cluster
   std::vector<TH1F *> hrsc; // reconstructed strip cluster
};

/*****************************************************************************/
void ClusterShapeExtractor::beginJob(const edm::EventSetup& es)
{
  // Get tracker geometry
  edm::ESHandle<TrackerGeometry>          tracker;
  es.get<TrackerDigiGeometryRecord>().get(tracker);
  theTracker =                            tracker.product();

  // Get magnetic field
  edm::ESHandle<MagneticField>           magneticField;
  es.get<IdealMagneticFieldRecord>().get(magneticField);
  theMagneticField =                     magneticField.product();

  // Get Lorentz angle for strips
  edm::ESHandle<SiStripLorentzAngle>   lorentzAngle;
  es.get<SiStripLorentzAngleRcd>().get(lorentzAngle);
  theLorentzAngle =                    lorentzAngle.product();

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
  hasSimInfo    = pset.getParameter<bool>("hasSimInfo"); 

  file = new TFile("clusterShape.root","RECREATE");
  file->cd();

  // pixel: subdet=0-1, ex=0-5, ey=0-13
  //        mx=, my=
  // strip: mw=0-6,
  //        condition: fs >= 2 && fs + mw - 1 <= ns - 2
  //        pw + orient * dw
}

/*****************************************************************************/
ClusterShapeExtractor::~ClusterShapeExtractor()
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
bool ClusterShapeExtractor::isSuitable(const PSimHit* simHit)
{
  // Is it outgoing ?
  DetId id = DetId(simHit->detUnitId());

  GlobalVector gvec = theTracker->idToDetUnit(id)->position() -
                      GlobalPoint(0,0,0);
  LocalVector  lvec = theTracker->idToDetUnit(id)->toLocal(gvec);
  LocalVector  ldir = simHit->exitPoint() - simHit->entryPoint();

  bool isOutgoing = (lvec.z()*ldir.z() > 0); 

  // Is it from a relevant process and particle
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

  // Is fast enough
  bool isFast = (simHit->momentumAtEntry().perp() > 0.050);

  return (isOutgoing && isRelevant && isFast);
}

/*****************************************************************************/
bool ClusterShapeExtractor::getOrientation
  (const SiStripRecHit2D& recHit, float& tangent, int& nstrips, bool& isBarrel)
{
  DetId id = recHit.geographicalId();

  const StripGeomDetUnit* stripDet =
      dynamic_cast<const StripGeomDetUnit*> (theTracker->idToDet(id));

  tangent  = stripDet->specificTopology().localPitch(LocalPoint(0.,0.,0.))/
             stripDet->surface().bounds().thickness();
  
  nstrips = stripDet->specificTopology().nstrips();
  
  if(stripDet->type().subDetector() == GeomDetEnumerators::TIB ||
     stripDet->type().subDetector() == GeomDetEnumerators::TOB)
  { 
    isBarrel = true;
    float perp0 = stripDet->toGlobal( Local3DPoint(0.,0.,0.) ).perp();
    float perp1 = stripDet->toGlobal( Local3DPoint(0.,0.,1.) ).perp();
    return (perp1 > perp0);
  }
  else
  { 
    isBarrel = false;
    float rot = stripDet->toGlobal( LocalVector (0.,0.,1.) ).z();
    float pos = stripDet->toGlobal( Local3DPoint(0.,0.,0.) ).z();
    return (rot * pos > 0);
  }
}

/*****************************************************************************/
void ClusterShapeExtractor::process(const SiStripRecHit2D* recHit,
  vector<PSimHit> simHits)
{
  int measuredWidth = recHit->cluster()->amplitudes().size();
  int   firstStrip  = recHit->cluster()->firstStrip();

  float tangent;
  int nstrips;
  bool isBarrel;
  bool isNormalOriented =
    getOrientation(*recHit, tangent, nstrips, isBarrel);

  // Cluster should have only one association
  if(simHits.size() == 1)
  {
    // Take first
    const PSimHit* simHit = &(simHits[0]);

    if(isSuitable(simHit)) // isBarrel
    {
      int orient = (isNormalOriented ? 1 : -1);

      LocalVector ldir = simHit->exitPoint() - simHit->entryPoint();
      float predictedWidth  = ldir.x() / (fabs(ldir.z()) * tangent);

      int   mw = measuredWidth;
      float pw = predictedWidth;

      DetId id = recHit->geographicalId();
      LocalPoint  lpos = recHit->localPosition();
      GlobalPoint gpos = theTracker->idToDet(id)->toGlobal(lpos);

      GlobalVector bfld = theMagneticField->inTesla(gpos);
      LocalVector Bfield = theTracker->idToDet(id)->toLocal(bfld);

      double theTanLorentzAnglePerTesla =
        theLorentzAngle->getLorentzAngle(id.rawId());

      float dir_x =  theTanLorentzAnglePerTesla * Bfield.y();
      float dir_z = -1.;
      float driftWidth = dir_x / (fabs(dir_z) * tangent);

      int fs = firstStrip;
      int ns = nstrips;

      float dw = driftWidth;

      if(fs >= 2 && fs + mw - 1 <= ns - 2)
      if(mw <= ewMax)
      {
        int i = mw;
        hssc[i]->Fill(pw + orient * dw);
      }
    }
  }
}

/*****************************************************************************/
void ClusterShapeExtractor::process
  (LocalVector ldir, const SiStripRecHit2D* recHit)
{
  int measuredWidth = recHit->cluster()->amplitudes().size();
  int   firstStrip  = recHit->cluster()->firstStrip();

  float tangent;
  int nstrips;
  bool isBarrel;
  bool isNormalOriented =
    getOrientation(*recHit, tangent, nstrips, isBarrel);

  // Cluster should have only one association
  // isBarrel
  {
    int orient = (isNormalOriented ? 1 : -1);

    float predictedWidth  = ldir.x() / (fabs(ldir.z()) * tangent);

    int   mw = measuredWidth;
    float pw = predictedWidth;

    DetId id = recHit->geographicalId();
    LocalPoint  lpos = recHit->localPosition();
    GlobalPoint gpos = theTracker->idToDet(id)->toGlobal(lpos);

    GlobalVector bfld = theMagneticField->inTesla(gpos);
    LocalVector Bfield = theTracker->idToDet(id)->toLocal(bfld);

    double theTanLorentzAnglePerTesla =
      theLorentzAngle->getLorentzAngle(id.rawId());

    float dir_x =  theTanLorentzAnglePerTesla * Bfield.y();
    float dir_z = -1.;
    float driftWidth = dir_x / (fabs(dir_z) * tangent);

    int fs = firstStrip;
    int ns = nstrips;

    float dw = driftWidth;

    if(fs >= 2 && fs + mw - 1 <= ns - 2)
    if(mw <= ewMax)
    {
      int i = mw;
      hrsc[i]->Fill(pw + orient * dw);
    }
  }
}

/*****************************************************************************/
void ClusterShapeExtractor::process(const SiPixelRecHit* recHit,
  vector<PSimHit> simHits)
{
  DetId id = recHit->geographicalId();
  const PixelGeomDetUnit* pixelDet =
    dynamic_cast<const PixelGeomDetUnit*> (theTracker->idToDet(id));

  ClusterShape theClusterShape;
  ClusterData data;
  theClusterShape.getExtra(*pixelDet, *recHit, data);

  // Cluster should be straight, complete and only one association
  if(data.isStraight && data.isComplete && simHits.size() == 1)
  {
    const PSimHit* simHit = &(simHits[0]);

    int recHitsSize = recHits_->get(id).second -
                      recHits_->get(id).first;

    if(isSuitable(simHit) && recHitsSize == 1)
    {
      int orient = (data.isNormalOriented ? 1 : -1);

      pair<float,float> move;

      LocalVector ldir = simHit->exitPoint() - simHit->entryPoint();
      move.first  =
        ldir.x() / (fabs(ldir.z()) * data.tangent.first ) * orient;
      move.second =
        ldir.y() / (fabs(ldir.z()) * data.tangent.second) * orient;

      int subdet = (data.isInBarrel ? 0 : 1);

      int ex = data.size.first;
      int ey = data.size.second;
      float mx = move.first;
      float my = move.second;

      if(ey < 0)
      { ey = -ey; my = -my; }

      if(ex <= exMax && ey >= 0. && ey <= eyMax)
      {
        int i = (subdet * (exMax + 1) + ex) * (eyMax + 1) + (ey);
        hspc[i]->Fill(mx,my);
      }
    }
  }
}

/*****************************************************************************/
void ClusterShapeExtractor::process
  (LocalVector ldir, const SiPixelRecHit* recHit)
{
  DetId id = recHit->geographicalId();
  const PixelGeomDetUnit* pixelDet =
    dynamic_cast<const PixelGeomDetUnit*> (theTracker->idToDet(id));

  ClusterShape theClusterShape;
  ClusterData data;
  theClusterShape.getExtra(*pixelDet, *recHit, data);

  // Cluster should be straight, complete and only one association
  if(data.isStraight && data.isComplete)
  {
    int orient = (data.isNormalOriented ? 1 : -1);

    pair<float,float> move;

    move.first  =
      ldir.x() / (fabs(ldir.z()) * data.tangent.first ) * orient;
    move.second =
      ldir.y() / (fabs(ldir.z()) * data.tangent.second) * orient;

    int subdet = (data.isInBarrel ? 0 : 1);

    int ex = data.size.first;
    int ey = data.size.second;
    float mx = move.first;
    float my = move.second;

    if(ey < 0)
    { ey = -ey; my = -my; }

    if(ex <= exMax && ey >= 0. && ey <= eyMax)
    {
      int i = (subdet * (exMax + 1) + ex) * (eyMax + 1) + (ey);
      hrpc[i]->Fill(mx,my);
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
  vector<edm::Handle<SiPixelRecHitCollection> > colls;
  ev.getManyByType(colls);

  for(vector<edm::Handle<SiPixelRecHitCollection> >::const_iterator
        coll = colls.begin(); coll!= colls.end(); coll++)
  {
    const SiPixelRecHitCollection* recHits = (*coll).product();
    recHits_ = recHits;

    nhits = recHits->size();

    for(  SiPixelRecHitCollection::const_iterator
          recHit = recHits->begin(); recHit!= recHits->end(); recHit++)
      process(&(*recHit), theHitAssociator->associateHit(*recHit));       
  }
  }

  // Strip hits
  {
  vector<edm::Handle<SiStripRecHit2DCollection> > colls;
  ev.getManyByType(colls);

  for(vector<edm::Handle<SiStripRecHit2DCollection> >::const_iterator
        coll = colls.begin(); coll!= colls.end(); coll++)
  {
    const SiStripRecHit2DCollection* recHits = (*coll).product();
    for(  SiStripRecHit2DCollection::const_iterator
          recHit = recHits->begin(); recHit!= recHits->end(); recHit++)
      process(&(*recHit), theHitAssociator->associateHit(*recHit));
  }
  }

  // Matched strip hits
  {
  vector<edm::Handle<SiStripMatchedRecHit2DCollection> > colls;
  ev.getManyByType(colls);

  for(vector<edm::Handle<SiStripMatchedRecHit2DCollection> >::const_iterator
        coll = colls.begin(); coll!= colls.end(); coll++)
  {
    const SiStripMatchedRecHit2DCollection* recHits = (*coll).product();
    for(  SiStripMatchedRecHit2DCollection::const_iterator
          recHit = recHits->begin(); recHit!= recHits->end(); recHit++)
    {
      process(&(*(recHit->monoHit())),
                theHitAssociator->associateHit(*(recHit->monoHit())));
      process(&(*(recHit->stereoHit())),
                theHitAssociator->associateHit(*(recHit->stereoHit())));
    }
  }
  }

  delete theHitAssociator;
}

/*****************************************************************************/
void ClusterShapeExtractor::analyzeRecHits
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
          process(ldir, pixelRecHit);
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

        pair<double,double> v;

        if(stripMatchedRecHit != 0)
        {
          process(ldir,stripMatchedRecHit->monoHit()  );
          process(ldir,stripMatchedRecHit->stereoHit());
        }

        if(stripProjectedRecHit != 0)
          process(ldir,&(stripProjectedRecHit->originalHit()));

        if(stripRecHit != 0)
          process(ldir,stripRecHit);
      }
    }
  }
}

/*****************************************************************************/
void ClusterShapeExtractor::analyze
  (const edm::Event& ev, const edm::EventSetup& es)
{
  if(hasSimInfo)
  {
    LogTrace("MinBiasTracking") << "[ClusterShape] analyze simHits";
    analyzeSimHits(ev, es);
  } 

  LogTrace("MinBiasTracking") << "[ClusterShape] analyze recHits";
  analyzeRecHits(ev,es);
}

DEFINE_FWK_MODULE(ClusterShapeExtractor);

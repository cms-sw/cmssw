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

#include "TROOT.h"
#include "TFile.h"
#include "TNtuple.h"

#include <utility>
#include <vector>
#include <fstream>
using namespace std;

/*****************************************************************************/
class ClusterShapeExtractor : public edm::EDAnalyzer
{
 public:
   explicit ClusterShapeExtractor(const edm::ParameterSet& pset);
   ~ClusterShapeExtractor();
   virtual void beginJob(const edm::EventSetup& es);
   virtual void analyze(const edm::Event& ev, const edm::EventSetup& es);
   virtual void endJob() { }

 private:
   pair<float,float> pointToPixel(DetId id, Local3DPoint point);
   void printCluster(const TrackingRecHit* recHit);
   bool isSuitable(const PSimHit* simHit);
   bool getOrientation(const SiStripRecHit2D& recHit, float& tangent, int& nstrips, bool& isBarrel);

   void processEnergyLoss
     (const SiPixelRecHit* recHit, vector<PSimHit> simHits);

   void process(const SiPixelRecHit*   recHit, vector<PSimHit> simHits);
   void process(const SiStripRecHit2D* recHit, vector<PSimHit> simHits);


   TFile * file;
   TNtuple * pixelShape;
   TNtuple * stripShape;

   const TrackerGeometry* theTracker;
   const MagneticField* theMagField;
   const SiPixelRecHitCollection* recHits_;

   edm::ParameterSet theConfig;
   int nhits;
   TrackerHitAssociator * theHitAssociator;
};

/*****************************************************************************/
void ClusterShapeExtractor::beginJob(const edm::EventSetup& es)
{
  // Get tracker geometry
  edm::ESHandle<TrackerGeometry> tracker;
  es.get<TrackerDigiGeometryRecord>().get(tracker);
  theTracker = tracker.product();

    // Get magnetic field
  edm::ESHandle<MagneticField> magField;
  es.get<IdealMagneticFieldRecord>().get(magField);
  theMagField = magField.product();
}

/*****************************************************************************/
ClusterShapeExtractor::ClusterShapeExtractor
  (const edm::ParameterSet& pset) : theConfig(pset)
{
  file = new TFile("clusterShape.root","RECREATE");
  file->cd();

  pixelShape = new TNtuple("pixelShape","pixelShape", "nh:subdet:ex:ey:mx:my");
  stripShape = new TNtuple("stripShape","stripShape", "mw:pw:fs:ns:dw:orient");
}

/*****************************************************************************/
ClusterShapeExtractor::~ClusterShapeExtractor()
{
  file->cd();

  pixelShape->Write();
  stripShape->Write();

  file->Close();
}

/*****************************************************************************/
pair<float,float> ClusterShapeExtractor::pointToPixel(DetId id, Local3DPoint point)
{
  const PixelGeomDetUnit* pixelDet =
    dynamic_cast<const PixelGeomDetUnit*> (theTracker->idToDet(id));

  float thickness = pixelDet->surface().bounds().thickness();
  float TanLorentzAnglePerTesla = 0.106;
  float Bfield = 4.;

  float dx = (thickness/2 + point.z()) * TanLorentzAnglePerTesla * Bfield;

  // drift to -z
  Local3DPoint drifted(point.x() + dx, point.y(), -thickness/2);

  return pixelDet->specificTopology().pixel(drifted);
}

/*****************************************************************************/
void ClusterShapeExtractor::printCluster(const TrackingRecHit* recHit)
{
  ofstream outFile("event/cluster.dat");

  DetId id = recHit->geographicalId();

  // recCluster
  const SiPixelRecHit* pixelRecHit =
    dynamic_cast<const SiPixelRecHit *>(recHit);

  SiPixelRecHit::ClusterRef const& cluster = pixelRecHit->cluster();
  vector<SiPixelCluster::Pixel> pixels = cluster->pixels();

  float x0 = 1e+9, x1=-1;
  float y0 = 1e+9, y1=-1;

  for(vector<SiPixelCluster::Pixel>::const_iterator
        pixel = pixels.begin(); pixel!= pixels.end(); pixel++)
  {
    if(pixel->x < x0) x0 = pixel->x;
    if(pixel->x > x1) x1 = pixel->x;

    if(pixel->y < y0) y0 = pixel->y;
    if(pixel->y > y1) y1 = pixel->y;
  }

  float d = 2.;
  float dx,dy;

  if(x1-x0 > y1-y0)
  { 
    dx = d; dy = dx + 1./2 * ((x1-x0) - (y1-y0));
  }
  else
  {
    dy = d; dx = dy + 1./2 * ((y1-y0) - (x1-x0));
  }

  // ranges
  outFile << "   " << x0-dx << " " << x1+dx
            << " " << y0-dy << " " << y1+dy << endl;

  // actual cluster
  for(vector<SiPixelCluster::Pixel>::const_iterator
        pixel = pixels.begin(); pixel!= pixels.end(); pixel++)
    outFile << "   " << pixel->x
              << " " << pixel->y
              << " " << pixel->adc / 135
              << endl;

  // all clusters in unit
  SiPixelRecHitCollection::range range = recHits_->get(id);
  for(SiPixelRecHitCollection::const_iterator rHit = range.first;
                                              rHit!= range.second; rHit++)
  {
    const SiPixelRecHit* pixelRecHit = 
      dynamic_cast<const SiPixelRecHit *>(&(*rHit));
  
    SiPixelRecHit::ClusterRef const& cluster = pixelRecHit->cluster();
    vector<SiPixelCluster::Pixel> pixels = cluster->pixels();
  
    for(vector<SiPixelCluster::Pixel>::const_iterator 
          pixel = pixels.begin(); pixel!= pixels.end(); pixel++)
      outFile << "   " << pixel->x
                << " " << pixel->y
                << " " << - pixel->adc/135
                << endl;
  }
  
  // simCluster
  vector<PSimHit> simHits = theHitAssociator->associateHit(*recHit);

  for(vector<PSimHit>::const_iterator
        simHit = simHits.begin(); simHit!= simHits.end(); simHit++)
  { 
    pair<float,float> entryPoint = pointToPixel(id,simHit->entryPoint());
    pair<float,float> exitPoint  = pointToPixel(id,simHit->exitPoint());

    outFile << "   " << entryPoint.first
              << " " << entryPoint.second
              << " " << exitPoint.first
              << " " << exitPoint.second
              << endl;
  }
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
  if(simHit->processType() == 2 || // fElectromagnetic
    (simHit->processType() == 3 && // fOptical
     simHit->particleType() != -100 &&
     simHit->particleType() != -101 &&
     simHit->particleType() != -102) ||
     simHit->processType() == 4)
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

    if(isSuitable(simHit) && isBarrel)
    {
      int orient = (isNormalOriented ? 1 : -1);

      LocalVector ldir = simHit->exitPoint() - simHit->entryPoint();
      float predictedWidth  = ldir.x() / (fabs(ldir.z()) * tangent);

      vector<float> result;
      result.push_back( measuredWidth); // mw
      result.push_back(predictedWidth); // pw

      DetId id = recHit->geographicalId();
      LocalPoint  lpos = recHit->localPosition();
      GlobalPoint gpos = theTracker->idToDet(id)->toGlobal(lpos);

      GlobalVector bfld = theMagField->inTesla(gpos);
      LocalVector Bfield = theTracker->idToDet(id)->toLocal(bfld);
      double theTanLorentzAnglePerTesla = 0.032;

      float dir_x =  theTanLorentzAnglePerTesla * Bfield.y();
      float dir_z = -1.;
      float driftWidth = dir_x / (fabs(dir_z) * tangent);

      result.push_back(firstStrip); // fs
      result.push_back(nstrips);    // ns

      result.push_back(driftWidth); // dw
      result.push_back(orient);     // orient

      stripShape->Fill(&result[0]);
    }
  }
}

/*****************************************************************************/
void ClusterShapeExtractor::processEnergyLoss
  (const SiPixelRecHit* recHit, vector<PSimHit> simHits)
{
  if(simHits.size() == 1)
  {
    const PSimHit* simHit = &(simHits[0]);

    vector<float> result;

    result.push_back(simHit->momentumAtEntry().mag());
    result.push_back(simHit->particleType());
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

      vector<float> result;
      result.push_back(nhits);
      result.push_back(subdet);
      result.push_back(data.size.first );
      result.push_back(data.size.second);
      result.push_back(move.first );
      result.push_back(move.second);

      // Infos needed for dE/dx calibration 
/*
      for(pixels)
      {
      result.push_back(simHit->momentumAtEntry().mag());           // p
      result.push_back(simHit->particleType());                    // id
      result.push_back(pixel->calculatedValues.partialLength * len);  // dx
      result.push_back(pixel->measuredValues.adc/135.);               // de,adc 
      result.push_back((hit.pixelDet->type().subDetector() ==
                         GeomDetEnumerators::PixelBarrel ? 0 : 1)); // type
      }
*/
  

/*
      if(data.size.first == 0 && data.size.second == 0 && fabs(move.second) > 6)
      {
        printCluster(recHit);
        cerr << " pixel cluster printed " << move.first
                                   << " " << move.second << endl;
        while(getchar() == 0);
      }
*/

      pixelShape->Fill(&result[0]);
    }
  }
}

/*****************************************************************************/
void ClusterShapeExtractor::analyze
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

DEFINE_FWK_MODULE(ClusterShapeExtractor);

/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*   Jan Kašpar (jan.kaspar@gmail.com) 
*
****************************************************************************/

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/TotemRPDetId/interface/TotemRPDetId.h"
#include "DataFormats/CTPPSReco/interface/TotemRPRecHit.h"
#include "DataFormats/CTPPSReco/interface/TotemRPUVPattern.h"

#include "Geometry/Records/interface/VeryForwardRealGeometryRecord.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/TotemRPGeometry.h"

#include "RecoCTPPS/TotemRPLocal/interface/FastLineRecognition.h"

//----------------------------------------------------------------------------------------------------

/**
 * \brief Class to recognize straight line tracks, based on optimized Hough trasform.
 *
 * The search is perfomed in global U,V coordinates (wrt. beam). In this way (some of)
 * the alignment corrections can be taken into account.
**/
class TotemRPUVPatternFinder : public edm::stream::EDProducer<>
{
  public:
    TotemRPUVPatternFinder(const edm::ParameterSet& conf);
    virtual ~TotemRPUVPatternFinder();
    virtual void produce(edm::Event& e, const edm::EventSetup& c);
  
  private:
    edm::InputTag tagRecHit;
    edm::EDGetTokenT<edm::DetSetVector<TotemRPRecHit> > detSetVectorTotemRPRecHitToken;

    unsigned int verbosity;

    /// minimal required number of active planes per projection to even start track recognition
    unsigned char minPlanesPerProjectionToSearch;

    /// minimal required number of active planes per projection to mark track candidate as fittable
    unsigned char minPlanesPerProjectionToFit;

    /// above this limit, planes are considered noisy
    unsigned int maxHitsPerPlaneToSearch;

    /// the line recognition algorithm
    FastLineRecognition *lrcgn;

    /// minimal weight of (Hough) cluster to accept it as candidate
    double threshold;

    /// maximal angle (in any projection) to mark candidate as fittable - controls track parallelity
    double max_a_toFit;

    /// exceptional settings, per RP and per projection
    std::vector<edm::ParameterSet> exceptionalSettings;

    edm::ESWatcher<VeryForwardRealGeometryRecord> geometryWatcher;

    /// executes line recognition in a projection
    void RecognizeAndSelect(TotemRPUVPattern::ProjectionType proj, double z0, double threshold,
      unsigned int planes_required,
      const edm::DetSetVector<TotemRPRecHit> &hits, edm::DetSet<TotemRPUVPattern> &patterns);
};

//----------------------------------------------------------------------------------------------------

using namespace std;
using namespace edm;

//----------------------------------------------------------------------------------------------------

TotemRPUVPatternFinder::TotemRPUVPatternFinder(const edm::ParameterSet& conf) :
  tagRecHit(conf.getParameter<edm::InputTag>("tagRecHit")),
  verbosity(conf.getUntrackedParameter<unsigned int>("verbosity", 0)),
  minPlanesPerProjectionToSearch(conf.getParameter<unsigned int>("minPlanesPerProjectionToSearch")),
  minPlanesPerProjectionToFit(conf.getParameter<unsigned int>("minPlanesPerProjectionToFit")),
  maxHitsPerPlaneToSearch(conf.getParameter<unsigned int>("maxHitsPerPlaneToSearch")),
  lrcgn(new FastLineRecognition(conf.getParameter<double>("clusterSize_a"), conf.getParameter<double>("clusterSize_b"))),
  threshold(conf.getParameter<double>("threshold")),
  max_a_toFit(conf.getParameter<double>("max_a_toFit")),
  exceptionalSettings(conf.getParameter< vector<ParameterSet> >("exceptionalSettings"))
{
  detSetVectorTotemRPRecHitToken = consumes<edm::DetSetVector<TotemRPRecHit> >(tagRecHit);

  produces<DetSetVector<TotemRPUVPattern>>();
}

//----------------------------------------------------------------------------------------------------

TotemRPUVPatternFinder::~TotemRPUVPatternFinder()
{
  delete lrcgn;
}

//----------------------------------------------------------------------------------------------------

void TotemRPUVPatternFinder::RecognizeAndSelect(TotemRPUVPattern::ProjectionType proj,
    double z0, double threshold_loc, unsigned int planes_required,
    const DetSetVector<TotemRPRecHit> &hits, DetSet<TotemRPUVPattern> &patterns)
{
  // run recognition
  DetSet<TotemRPUVPattern> newPatterns;
  lrcgn->GetPatterns(hits, z0, threshold_loc, newPatterns);
  
  // set pattern properties and copy to the global pattern collection
  for (auto &p : newPatterns)
  {
    p.setProjection(proj);

    p.setFittable(true);

    set<unsigned int> planes;
    for (const auto &ds : p.getHits())
        planes.insert(TotemRPDetId::rawToDecId(ds.detId()) % 10);

    if (planes.size() < planes_required)
      p.setFittable(false);
    
    if (fabs(p.getA()) > max_a_toFit)
      p.setFittable(false);

    patterns.push_back(p);
  }
}

//----------------------------------------------------------------------------------------------------

void TotemRPUVPatternFinder::produce(edm::Event& event, const edm::EventSetup& es)
{
  if (verbosity > 5)
    printf(">> TotemRPUVPatternFinder::produce (%u:%llu)\n", event.id().run(), event.id().event());

  // geometry
  ESHandle<TotemRPGeometry> geometry;
  es.get<VeryForwardRealGeometryRecord>().get(geometry);
  if (geometryWatcher.check(es))
    lrcgn->ResetGeometry(geometry.product());
  
  // get input
  edm::Handle< edm::DetSetVector<TotemRPRecHit> > input;
  event.getByToken(detSetVectorTotemRPRecHitToken, input);

  // prepare output
  DetSetVector<TotemRPUVPattern> patternsVector;
  
  // split input per RP and per U/V projection
  struct RPData
  {
    DetSetVector<TotemRPRecHit> hits_U, hits_V;
    map<uint8_t, uint16_t> planeOccupancy_U, planeOccupancy_V;
  };
  map<unsigned int, RPData> rpData;

  for (auto &ids : *input)
  {
    unsigned int detId = TotemRPDetId::rawToDecId(ids.detId());
    unsigned int rpId = TotemRPDetId::rpOfDet(detId);
    unsigned int plane = detId % 10;
    bool uDir = TotemRPDetId::isStripsCoordinateUDirection(detId);

    RPData &data = rpData[rpId];

    for (auto &h : ids)
    {
      if (uDir)
      {
        auto &ods = data.hits_U.find_or_insert(ids.detId());
        ods.push_back(h);
        data.planeOccupancy_U[plane]++;
      } else {
        auto &ods = data.hits_V.find_or_insert(ids.detId());
        ods.push_back(h);
        data.planeOccupancy_V[plane]++;
      }
    }
  }

  // track recognition pot by pot
  for (auto it : rpData)
  {
    unsigned int rpId = it.first;
    RPData &data = it.second;

    // merge default and exceptional settings (if available)
    unsigned int minPlanesPerProjectionToFit_U = minPlanesPerProjectionToFit;
    unsigned int minPlanesPerProjectionToFit_V = minPlanesPerProjectionToFit;
    double threshold_U = threshold;
    double threshold_V = threshold;

    for (auto ps : exceptionalSettings)
    {
      unsigned int setId = ps.getParameter<unsigned int>("rpId");
      if (setId == rpId)
      {
        minPlanesPerProjectionToFit_U = ps.getParameter<unsigned int>("minPlanesPerProjectionToFit_U");
        minPlanesPerProjectionToFit_V = ps.getParameter<unsigned int>("minPlanesPerProjectionToFit_V");
        threshold_U = ps.getParameter<double>("threshold_U");
        threshold_V = ps.getParameter<double>("threshold_V");
      }
    }

    auto &uColl = data.planeOccupancy_U;
    auto &vColl = data.planeOccupancy_V;

    if (verbosity > 5)
    {
      printf("\tRP %u\n", rpId);
      printf("\t\tall planes: u = %lu, v = %lu\n", uColl.size(), vColl.size());
    }

    // count planes with clean data (no showers, noise, ...)
    unsigned int uPlanes = 0, vPlanes = 0;
    for (auto pit : uColl)
      if (pit.second <= maxHitsPerPlaneToSearch)
        uPlanes++;

    for (auto pit : vColl)
      if (pit.second <= maxHitsPerPlaneToSearch)
        vPlanes++;

    if (verbosity > 5)
      printf("\t\tplanes with clean data: u = %u, v = %i\n", uPlanes, vPlanes);

    // discard RPs with too few reasonable planes
    if (uPlanes < minPlanesPerProjectionToSearch || vPlanes < minPlanesPerProjectionToSearch)
      continue;

    // prepare data containers
    DetSet<TotemRPUVPattern> &patterns = patternsVector.find_or_insert(rpId);

    // "typical" z0 for the RP
    double z0 = geometry->GetRPDevice(rpId)->translation().z();

    // u then v recognition
    if (verbosity > 5)
      printf("\t\tu recognition\n");
    RecognizeAndSelect(TotemRPUVPattern::projU, z0, threshold_U, minPlanesPerProjectionToFit_U, data.hits_U, patterns);

    if (verbosity > 5)
      printf("\t\tv recognition\n");
    RecognizeAndSelect(TotemRPUVPattern::projV, z0, threshold_V, minPlanesPerProjectionToFit_V, data.hits_V, patterns);

    if (verbosity > 5)
    {
      printf("\t\tpatterns:\n");
      for (const auto &p : patterns)
      {
        unsigned int n_hits = 0;
        for (auto &hds : p.getHits())
          n_hits += hds.size();

        printf("\t\t\tproj = %s, a = %.3f, b = %.3f, w = %.3f, fittable = %i, hits = %u\n",
          (p.getProjection() == TotemRPUVPattern::projU) ? "U" : "V",
          p.getA(), p.getB(), p.getW(), p.getFittable(), n_hits);
      }
    }
  }
 
  // save output
  event.put(make_unique<DetSetVector<TotemRPUVPattern>>(patternsVector));
}
 
//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(TotemRPUVPatternFinder);

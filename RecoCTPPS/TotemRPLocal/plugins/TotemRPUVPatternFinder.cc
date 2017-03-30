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
#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"
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

    virtual void produce(edm::Event& e, const edm::EventSetup& c) override;
  
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

    /// block of (exceptional) settings for 1 RP
    struct RPSettings
    {
      unsigned char minPlanesPerProjectionToFit_U, minPlanesPerProjectionToFit_V;
      double threshold_U, threshold_V;
    };

    /// exceptional settings: RP Id --> settings
    std::map<unsigned int, RPSettings> exceptionalSettings;

    edm::ESWatcher<VeryForwardRealGeometryRecord> geometryWatcher;

    /// executes line recognition in a projection
    void recognizeAndSelect(TotemRPUVPattern::ProjectionType proj, double z0, double threshold,
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
  max_a_toFit(conf.getParameter<double>("max_a_toFit"))
{
  for (const auto &ps : conf.getParameter< vector<ParameterSet> >("exceptionalSettings"))
  {
    unsigned int rpId = ps.getParameter<unsigned int>("rpId");

    RPSettings settings;
    settings.minPlanesPerProjectionToFit_U = ps.getParameter<unsigned int>("minPlanesPerProjectionToFit_U");
    settings.minPlanesPerProjectionToFit_V = ps.getParameter<unsigned int>("minPlanesPerProjectionToFit_V");
    settings.threshold_U = ps.getParameter<double>("threshold_U");
    settings.threshold_V = ps.getParameter<double>("threshold_V");

    exceptionalSettings[rpId] = settings;
  }

  detSetVectorTotemRPRecHitToken = consumes<edm::DetSetVector<TotemRPRecHit> >(tagRecHit);

  produces<DetSetVector<TotemRPUVPattern>>();
}

//----------------------------------------------------------------------------------------------------

TotemRPUVPatternFinder::~TotemRPUVPatternFinder()
{
  delete lrcgn;
}

//----------------------------------------------------------------------------------------------------

void TotemRPUVPatternFinder::recognizeAndSelect(TotemRPUVPattern::ProjectionType proj,
    double z0, double threshold_loc, unsigned int planes_required,
    const DetSetVector<TotemRPRecHit> &hits, DetSet<TotemRPUVPattern> &patterns)
{
  // run recognition
  DetSet<TotemRPUVPattern> newPatterns;
  lrcgn->getPatterns(hits, z0, threshold_loc, newPatterns);
  
  // set pattern properties and copy to the global pattern collection
  for (auto &p : newPatterns)
  {
    p.setProjection(proj);

    p.setFittable(true);

    set<unsigned int> planes;
    for (const auto &ds : p.getHits())
        planes.insert(TotemRPDetId(ds.detId()).plane());

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
    LogVerbatim("TotemRPUVPatternFinder")
      << ">> TotemRPUVPatternFinder::produce " << event.id().run() << ":" << event.id().event();

  // geometry
  ESHandle<TotemRPGeometry> geometry;
  es.get<VeryForwardRealGeometryRecord>().get(geometry);
  if (geometryWatcher.check(es))
    lrcgn->resetGeometry(geometry.product());
  
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
    TotemRPDetId detId(ids.detId());
    unsigned int plane = detId.plane();
    bool uDir = detId.isStripsCoordinateUDirection();

    CTPPSDetId rpId = detId.getRPId();

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
    CTPPSDetId rpId(it.first);
    RPData &data = it.second;

    // merge default and exceptional settings (if available)
    unsigned int minPlanesPerProjectionToFit_U = minPlanesPerProjectionToFit;
    unsigned int minPlanesPerProjectionToFit_V = minPlanesPerProjectionToFit;
    double threshold_U = threshold;
    double threshold_V = threshold;
    
    auto setIt = exceptionalSettings.find(rpId);
    if (setIt != exceptionalSettings.end())
    {
      minPlanesPerProjectionToFit_U = setIt->second.minPlanesPerProjectionToFit_U;
      minPlanesPerProjectionToFit_V = setIt->second.minPlanesPerProjectionToFit_V;
      threshold_U = setIt->second.threshold_U;
      threshold_V = setIt->second.threshold_V;
    }

    auto &uColl = data.planeOccupancy_U;
    auto &vColl = data.planeOccupancy_V;

    if (verbosity > 5)
    {
      LogVerbatim("TotemRPUVPatternFinder")
        << "\tRP " << rpId
        << "\n\t\tall planes: u = " << uColl.size() << ", v = " << vColl.size();
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
      LogVerbatim("TotemRPUVPatternFinder") << "\t\tplanes with clean data: u = " << uPlanes << ", v = " << vPlanes;

    // discard RPs with too few reasonable planes
    if (uPlanes < minPlanesPerProjectionToSearch || vPlanes < minPlanesPerProjectionToSearch)
      continue;

    // prepare data containers
    DetSet<TotemRPUVPattern> &patterns = patternsVector.find_or_insert(rpId);

    // "typical" z0 for the RP
    double z0 = geometry->GetRPDevice(rpId)->translation().z();

    // u then v recognition
    recognizeAndSelect(TotemRPUVPattern::projU, z0, threshold_U, minPlanesPerProjectionToFit_U, data.hits_U, patterns);

    recognizeAndSelect(TotemRPUVPattern::projV, z0, threshold_V, minPlanesPerProjectionToFit_V, data.hits_V, patterns);

    if (verbosity > 5)
    {
      LogVerbatim("TotemRPUVPatternFinder") << "\t\tpatterns:";
      for (const auto &p : patterns)
      {
        unsigned int n_hits = 0;
        for (auto &hds : p.getHits())
          n_hits += hds.size();
      
        LogVerbatim("TotemRPUVPatternFinder")
          << "\t\t\tproj = " << ((p.getProjection() == TotemRPUVPattern::projU) ? "U" : "V")
          << ", a = " << p.getA()
          << ", b = " << p.getB()
          << ", w = " << p.getW()
          << ", fittable = " << p.getFittable()
          << ", hits = " << n_hits;
      }
    }
  }
 
  // save output
  event.put(make_unique<DetSetVector<TotemRPUVPattern>>(patternsVector));
}
 
//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(TotemRPUVPatternFinder);

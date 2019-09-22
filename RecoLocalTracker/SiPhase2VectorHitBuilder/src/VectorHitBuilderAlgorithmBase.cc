#include "RecoLocalTracker/SiPhase2VectorHitBuilder/interface/VectorHitBuilderAlgorithmBase.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "RecoLocalTracker/Records/interface/TkPhase2OTCPERecord.h"
#include "RecoLocalTracker/Phase2TrackerRecHits/interface/Phase2StripCPE.h"

#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"

VectorHitBuilderAlgorithmBase::VectorHitBuilderAlgorithmBase(const edm::ParameterSet& conf) :
  nMaxVHforeachStack(conf.getParameter<int>("maxVectorHitsInAStack")),
  barrelCut(conf.getParameter< std::vector< double > >("BarrelCut")),
  endcapCut(conf.getParameter< std::vector< double > >("EndcapCut")),
  cpeTag_(conf.getParameter<edm::ESInputTag>("CPE"))
{}

void VectorHitBuilderAlgorithmBase::initialize(const edm::EventSetup& es)
{
  //FIXME:ask Vincenzo
  /*
  uint32_t tk_cache_id = es.get<TrackerDigiGeometryRecord>().cacheIdentifier();
  uint32_t c_cache_id = es.get<TkPhase2OTCPERecord>().cacheIdentifier();

  if(tk_cache_id != tracker_cache_id) {
    es.get<TrackerDigiGeometryRecord>().get(tracker);
    tracker_cache_id = tk_cache_id;
  }
  if(c_cache_id != cpe_cache_id) {
    es.get<TkPhase2OTCPERecord>().get(matcherTag, matcher);
    es.get<TkPhase2OTCPERecord>().get(cpeTag, cpe);
    cpe_cache_id = c_cache_id;
  }
  */

  // get the geometry and topology
  edm::ESHandle< TrackerGeometry > geomHandle;
  es.get< TrackerDigiGeometryRecord >().get( geomHandle );
  initTkGeom(geomHandle);

  edm::ESHandle< TrackerTopology > tTopoHandle;
  es.get< TrackerTopologyRcd >().get(tTopoHandle);
  initTkTopo(tTopoHandle);

  // load the cpe via the eventsetup
  edm::ESHandle<ClusterParameterEstimator<Phase2TrackerCluster1D> > cpeHandle;
  es.get<TkPhase2OTCPERecord>().get(cpeTag_, cpeHandle);
  initCpe(cpeHandle.product());
}

void VectorHitBuilderAlgorithmBase::initTkGeom(edm::ESHandle< TrackerGeometry > tkGeomHandle){
  theTkGeom = tkGeomHandle.product();
}
void VectorHitBuilderAlgorithmBase::initTkTopo(edm::ESHandle< TrackerTopology > tkTopoHandle){
  theTkTopo = tkTopoHandle.product();
}
void VectorHitBuilderAlgorithmBase::initCpe(const ClusterParameterEstimator<Phase2TrackerCluster1D>* cpeProd){
  cpe = cpeProd;
}

double VectorHitBuilderAlgorithmBase::computeParallaxCorrection(const PixelGeomDetUnit*& geomDetUnit_low, const Point3DBase<float, LocalTag>& lPosClu_low, 
                                                                const PixelGeomDetUnit*& geomDetUnit_upp, const Point3DBase<float, LocalTag>& lPosClu_upp){
  double parallCorr = 0.0;
  Global3DPoint origin(0,0,0);
  Global3DPoint gPosClu_low = geomDetUnit_low->surface().toGlobal(lPosClu_low);
  GlobalVector gV = gPosClu_low - origin;
  LogTrace("VectorHitsBuilderValidation") << " global vector passing to the origin:" << gV;

  LocalVector lV = geomDetUnit_low->surface().toLocal(gV);
  LogTrace("VectorHitsBuilderValidation") << " local vector passing to the origin (in low sor):" << lV;
  LocalVector lV_norm = lV/lV.z();
  LogTrace("VectorHitsBuilderValidation") << " normalized local vector passing to the origin (in low sor):" << lV_norm;

  Global3DPoint gPosClu_upp = geomDetUnit_upp->surface().toGlobal(lPosClu_upp);
  Local3DPoint lPosClu_uppInLow = geomDetUnit_low->surface().toLocal(gPosClu_upp);
  parallCorr = lV_norm.x() * lPosClu_uppInLow.z();

  return parallCorr;
}

void VectorHitBuilderAlgorithmBase::printClusters(const edmNew::DetSetVector<Phase2TrackerCluster1D>& clusters){

  int nCluster = 0;
  int numberOfDSV = 0;
  edmNew::DetSetVector<Phase2TrackerCluster1D>::const_iterator DSViter;
  for( DSViter = clusters.begin() ; DSViter != clusters.end(); DSViter++){

    ++numberOfDSV;

    // Loop over the clusters in the detector unit
    for (edmNew::DetSet< Phase2TrackerCluster1D >::const_iterator clustIt = DSViter->begin(); clustIt != DSViter->end(); ++clustIt) {

      nCluster++;

      // get the detector unit's id
      const GeomDetUnit* geomDetUnit(theTkGeom->idToDetUnit(DSViter->detId()));
      if (!geomDetUnit) return;

      printCluster(geomDetUnit, clustIt);

    }
  }
  LogDebug("VectorHitBuilder") << " Number of input clusters: " << nCluster << std::endl;

}


void VectorHitBuilderAlgorithmBase::printCluster(const GeomDet* geomDetUnit, const Phase2TrackerCluster1D* clustIt){

  if (!geomDetUnit) return;
  const PixelGeomDetUnit* pixelGeomDetUnit = dynamic_cast< const PixelGeomDetUnit* >(geomDetUnit);
  const PixelTopology& topol = pixelGeomDetUnit->specificTopology();
  if (!pixelGeomDetUnit) return;

  unsigned int layer = theTkTopo->layer(geomDetUnit->geographicalId());
  unsigned int module = theTkTopo->module(geomDetUnit->geographicalId());
  LogTrace("VectorHitBuilder") << "Layer:" << layer << " and DetId: " << geomDetUnit->geographicalId().rawId() << std::endl;
  TrackerGeometry::ModuleType mType = theTkGeom->getDetectorType(geomDetUnit->geographicalId());
  if (mType == TrackerGeometry::ModuleType::Ph2PSP) 
    LogTrace("VectorHitBuilder") << "Pixel cluster (module:" << module << ") " << std::endl;
  else if (mType == TrackerGeometry::ModuleType::Ph2SS || mType == TrackerGeometry::ModuleType::Ph2PSS) 
    LogTrace("VectorHitBuilder") << "Strip cluster (module:" << module << ") " << std::endl;
  else LogTrace("VectorHitBuilder") << "no module?!" << std::endl;
  LogTrace("VectorHitBuilder") << "with pitch:" << topol.pitch().first << " , " << topol.pitch().second << std::endl;
  LogTrace("VectorHitBuilder") << " and width:" << pixelGeomDetUnit->surface().bounds().width() << " , lenght:" << pixelGeomDetUnit->surface().bounds().length() << std::endl;


  auto && lparams = cpe->localParameters( *clustIt, *pixelGeomDetUnit );
  Global3DPoint gparams = pixelGeomDetUnit->surface().toGlobal(lparams.first);

  LogTrace("VectorHitBuilder") << "\t global pos " << gparams << std::endl;
  LogTrace("VectorHitBuilder") << "\t local  pos " << lparams.first << "with err " << lparams.second << std::endl;
  LogTrace("VectorHitBuilder") << std::endl;

  return;
}

void VectorHitBuilderAlgorithmBase::loadDetSetVector( std::map< DetId,std::vector<VectorHit> >& theMap, edmNew::DetSetVector<VectorHit>& theCollection ) const{

  std::map<DetId,std::vector<VectorHit> >::const_iterator it = theMap.begin();
  std::map<DetId,std::vector<VectorHit> >::const_iterator lastDet = theMap.end();
  for( ; it != lastDet ; ++it ) {
    edmNew::DetSetVector<VectorHit>::FastFiller vh_col(theCollection, it->first);
    std::vector<VectorHit>::const_iterator vh_it = it->second.begin();
    std::vector<VectorHit>::const_iterator vh_end = it->second.end();
    for( ; vh_it != vh_end ; ++vh_it)  {
      vh_col.push_back(*vh_it);
    }
  }

}

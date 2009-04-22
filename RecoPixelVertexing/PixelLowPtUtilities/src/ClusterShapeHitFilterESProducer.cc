#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShapeHitFilterESProducer.h"

//#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilterFactory.h"

#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShapeHitFilter.h"

#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"

#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

/*****************************************************************************/
ClusterShapeHitFilterESProducer::ClusterShapeHitFilterESProducer
  (const edm::ParameterSet& iConfig)
{
std::cerr << " HitFilterESP" << std::endl;
  componentName = iConfig.getParameter<std::string>("ComponentName");
  
  filterPset = iConfig.getParameter<edm::ParameterSet>("filterPset");
  componentType = filterPset.getParameter<std::string>("ComponentType");
  
  edm::LogInfo("ClusterShapeHitFilterESProducer")
    << "configured to produce: " << componentType
    << " with name: "            << componentName;
      
  setWhatProduced(this, componentName);
}


/*****************************************************************************/
ClusterShapeHitFilterESProducer::~ClusterShapeHitFilterESProducer
  ()
{
}

/*****************************************************************************/
ClusterShapeHitFilterESProducer::ReturnType
ClusterShapeHitFilterESProducer::produce
   (const CkfComponentsRecord &iRecord)
{
  using namespace edm::es;
  edm::LogInfo("ClusterShapeHitFilterESProducer")
    << "producing: " << componentName
    << " of type: "  << componentType;

  // Retrieve magnetic field
  edm::ESHandle<MagneticField> field;
  iRecord.getRecord<TrackingComponentsRecord>().getRecord<IdealMagneticFieldRecord>().get(field);

  // Retrieve geometry
  edm::ESHandle<GlobalTrackingGeometry> geo;
  iRecord.getRecord<TrackingComponentsRecord>().getRecord<GlobalTrackingGeometryRecord>().get(geo);

  // Retrieve pixel Lorentz
  edm::ESHandle<SiPixelLorentzAngle> pixel;
  iRecord.getRecord<TkPixelCPERecord>().getRecord<SiPixelLorentzAngleRcd>().get(pixel);

  // Retrieve strip Lorentz
  edm::ESHandle<SiStripLorentzAngle> strip;
  iRecord.getRecord<TkStripCPERecord>().getRecord<SiStripLorentzAngleRcd>().get(strip);

  // Produce the filter using the plugin factory
  ClusterShapeHitFilterESProducer::ReturnType
    aFilter(new ClusterShapeHitFilter(  geo.product(),
                                      field.product(),
                                      pixel.product(),
                                      strip.product()));

  return aFilter;
}

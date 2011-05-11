#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShapeHitFilterESProducer.h"

#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShapeHitFilter.h"

#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"

#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

/*****************************************************************************/
ClusterShapeHitFilterESProducer::ClusterShapeHitFilterESProducer
  (const edm::ParameterSet& iConfig)
{
  
  std::string componentName = iConfig.getParameter<std::string>("ComponentName");
  
  edm::LogInfo("ClusterShapeHitFilterESProducer")
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
(const ClusterShapeHitFilter::Record &iRecord)
{
  using namespace edm::es;

  // Retrieve magnetic field
  edm::ESHandle<MagneticField> field;
  iRecord.getRecord<TrackingComponentsRecord>().getRecord<IdealMagneticFieldRecord>().get(field);
  //  iRecord.getRecord<IdealMagneticFieldRecord>().get(field);
  //iRecord.get(field);

  // Retrieve geometry
  edm::ESHandle<GlobalTrackingGeometry> geo;
  iRecord.getRecord<TrackingComponentsRecord>().getRecord<GlobalTrackingGeometryRecord>().get(geo);
  //iRecord.getRecord<GlobalTrackingGeometryRecord>().get(geo);

  // Retrieve pixel Lorentz
  edm::ESHandle<SiPixelLorentzAngle> pixel;
  iRecord.getRecord<TkPixelCPERecord>().getRecord<SiPixelLorentzAngleRcd>().get(pixel);
  //iRecord.getRecord<SiPixelLorentzAngleRcd>().get(pixel);

  // Retrieve strip Lorentz
  edm::ESHandle<SiStripLorentzAngle> strip;
  iRecord.getRecord<TkStripCPERecord>().getRecord<SiStripLorentzAngleDepRcd>().get(strip);
  //iRecord.getRecord<SiStripLorentzAngleRcd>().get(strip);

  // Produce the filter using the plugin factory
  ClusterShapeHitFilterESProducer::ReturnType
    aFilter(new ClusterShapeHitFilter(  geo.product(),
                                      field.product(),
                                      pixel.product(),
                                      strip.product()));

  return aFilter;
}

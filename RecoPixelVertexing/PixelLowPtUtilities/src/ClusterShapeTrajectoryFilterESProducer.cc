#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShapeTrajectoryFilterESProducer.h"

#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilterFactory.h"

#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShapeTrajectoryFilter.h"

#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"

#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

/*****************************************************************************/
ClusterShapeTrajectoryFilterESProducer::ClusterShapeTrajectoryFilterESProducer
  (const edm::ParameterSet& iConfig)
{
  componentName = iConfig.getParameter<std::string>("ComponentName");
  
  filterPset = iConfig.getParameter<edm::ParameterSet>("filterPset");
  componentType = filterPset.getParameter<std::string>("ComponentType");
  
  edm::LogInfo("ClusterShapeTrajectoryFilterESProducer")
    << "configured to produce: " << componentType
    << " with name: "            << componentName;
      
  setWhatProduced(this, componentName);
}


/*****************************************************************************/
ClusterShapeTrajectoryFilterESProducer::~ClusterShapeTrajectoryFilterESProducer
  ()
{
}

/*****************************************************************************/
ClusterShapeTrajectoryFilterESProducer::ReturnType
ClusterShapeTrajectoryFilterESProducer::produce
   (const CkfComponentsRecord &iRecord)
{
  using namespace edm::es;
  edm::LogInfo("ClusterShapeTrajectoryFilterESProducer")
    << "producing: " << componentName
    << " of type: "  << componentType;

  // Retrieve magentic field
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
  ClusterShapeTrajectoryFilterESProducer::ReturnType
    aFilter(new ClusterShapeTrajectoryFilter(  geo.product(),
                                             field.product(),
                                             pixel.product(),
                                             strip.product()));
  return aFilter;
}

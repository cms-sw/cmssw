#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShapeTrajectoryFilterESProducer.h"

#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilterFactory.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShapeTrajectoryFilter.h"

#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

ClusterShapeTrajectoryFilterESProducer::ClusterShapeTrajectoryFilterESProducer(const edm::ParameterSet& iConfig)
{
  componentName = iConfig.getParameter<std::string>("ComponentName");
  
  
  filterPset = iConfig.getParameter<edm::ParameterSet>("filterPset");
  componentType = filterPset.getParameter<std::string>("ComponentType");
  
  edm::LogInfo("ClusterShapeTrajectoryFilterESProducer")<<"configured to produce: "<<componentType
					    <<" with name: "<<componentName;
      
  setWhatProduced(this, componentName);
}


ClusterShapeTrajectoryFilterESProducer::~ClusterShapeTrajectoryFilterESProducer(){}

ClusterShapeTrajectoryFilterESProducer::ReturnType
ClusterShapeTrajectoryFilterESProducer::produce
   (const TrackingComponentsRecord &iRecord)
{
   using namespace edm::es;
   edm::LogInfo("ClusterShapeTrajectoryFilterESProducer")<<"producing: "<<componentName<<" of type: "<<componentType;

   //retrieve magentic fiedl
   edm::ESHandle<MagneticField> field;
   iRecord.getRecord<IdealMagneticFieldRecord>().get(field);

   //retrieve geometry
   edm::ESHandle<GlobalTrackingGeometry> geo;
   iRecord.getRecord<GlobalTrackingGeometryRecord>().get(geo);

   //produce the filter using the plugin factory
   ClusterShapeTrajectoryFilterESProducer::ReturnType aFilter(new ClusterShapeTrajectoryFilter(geo.product(),
             field.product()));
             //  filterPset.getParameter<int>("Mode"));
   return aFilter ;
}

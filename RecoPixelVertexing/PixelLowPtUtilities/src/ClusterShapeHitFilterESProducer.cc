#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShapeHitFilterESProducer.h"

#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShapeHitFilter.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/ClusterChargeCut.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"


#include "FWCore/MessageLogger/interface/MessageLogger.h"

/*****************************************************************************/
ClusterShapeHitFilterESProducer::ClusterShapeHitFilterESProducer
  (const edm::ParameterSet& iConfig):
  pixelShapeFile(iConfig.getParameter<std::string>("PixelShapeFile")),
  pixelShapeFileL1(iConfig.getParameter<std::string>("PixelShapeFileL1"))
{
  
  std::string componentName = iConfig.getParameter<std::string>("ComponentName");
  minGoodPixelCharge_= 0,
  minGoodStripCharge_ = (clusterChargeCut(iConfig));
  cutOnPixelCharge_ = false;
  cutOnStripCharge_ = minGoodStripCharge_>0;
  cutOnPixelShape_ = (iConfig.exists("doPixelShapeCut") ? iConfig.getParameter<bool>("doPixelShapeCut") : true);
  cutOnStripShape_ = (iConfig.exists("doStripShapeCut") ? iConfig.getParameter<bool>("doStripShapeCut") : true);

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

  // get all from SiStripLorentzAngle (why not!)

  // Retrieve magnetic field
  edm::ESHandle<MagneticField> field;
  iRecord.getRecord<TkStripCPERecord>().getRecord<IdealMagneticFieldRecord>().get(field);

  // Retrieve geometry
  edm::ESHandle<TrackerGeometry> geo;
  iRecord.getRecord<TkStripCPERecord>().getRecord<TrackerDigiGeometryRecord>().get(geo);
  // Retrieve topology
  edm::ESHandle<TrackerTopology> topol;
  iRecord.getRecord<TrackerTopologyRcd>().get(topol);


  // Retrieve pixel Lorentz
  edm::ESHandle<SiPixelLorentzAngle> pixel;
  iRecord.getRecord<TkPixelCPERecord>().getRecord<SiPixelLorentzAngleRcd>().get(pixel);

  // Retrieve strip Lorentz
  edm::ESHandle<SiStripLorentzAngle> strip;
  iRecord.getRecord<TkStripCPERecord>().getRecord<SiStripLorentzAngleDepRcd>().get(strip);
 


  // Produce the filter using the plugin factory
  ClusterShapeHitFilterESProducer::ReturnType
    aFilter(new ClusterShapeHitFilter(  geo.product(),
                                      topol.product(),
                                      field.product(),
                                      pixel.product(),
                                      strip.product(),
                                      pixelShapeFile,
                                      pixelShapeFileL1
                                     ));

  aFilter->setShapeCuts(cutOnPixelShape_, cutOnStripShape_);
  aFilter->setChargeCuts(cutOnPixelCharge_, minGoodPixelCharge_, cutOnStripCharge_,
    minGoodStripCharge_);
  return aFilter;
}

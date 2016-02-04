#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEGenericESProducer.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEGeneric.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"



#include <string>
#include <memory>

using namespace edm;

PixelCPEGenericESProducer::PixelCPEGenericESProducer(const edm::ParameterSet & p) 
{
  std::string myname = p.getParameter<std::string>("ComponentName");
  pset_ = p;
  setWhatProduced(this,myname);
}

PixelCPEGenericESProducer::~PixelCPEGenericESProducer() {}

boost::shared_ptr<PixelClusterParameterEstimator>
PixelCPEGenericESProducer::produce(const TkPixelCPERecord & iRecord){ 

  ESHandle<MagneticField> magfield;
  iRecord.getRecord<IdealMagneticFieldRecord>().get( magfield );

  edm::ESHandle<TrackerGeometry> pDD;
  iRecord.getRecord<TrackerDigiGeometryRecord>().get( pDD );

  ESHandle<SiPixelLorentzAngle> lorentzAngle;
  iRecord.getRecord<SiPixelLorentzAngleRcd>().get(lorentzAngle );
	
	ESHandle<SiPixelCPEGenericErrorParm> genErrorParm;
	iRecord.getRecord<SiPixelCPEGenericErrorParmRcd>().get(genErrorParm);

	ESHandle<SiPixelTemplateDBObject> templateDBobject;
	iRecord.getRecord<SiPixelTemplateDBObjectESProducerRcd>().get(templateDBobject);

  cpe_  = boost::shared_ptr<PixelClusterParameterEstimator>(new PixelCPEGeneric(pset_,magfield.product(),lorentzAngle.product(),genErrorParm.product(),templateDBobject.product()) );
	//ToDo? Replace blah.product() with ESHandle
	
  return cpe_;
}



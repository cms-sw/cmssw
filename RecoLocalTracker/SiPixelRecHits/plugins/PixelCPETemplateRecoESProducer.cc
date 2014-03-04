#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPETemplateRecoESProducer.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPETemplateReco.h"
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

PixelCPETemplateRecoESProducer::PixelCPETemplateRecoESProducer(const edm::ParameterSet & p) 
{
  std::string myname = p.getParameter<std::string>("ComponentName");
  DoLorentz_ = p.getParameter<bool>("DoLorentz"); // True when LA from alignment is used

  pset_ = p;
  setWhatProduced(this,myname);
  //std::cout<<" from ES Producer Templates "<<myname<<" "<<DoLorentz_<<std::endl;
}

PixelCPETemplateRecoESProducer::~PixelCPETemplateRecoESProducer() {}

boost::shared_ptr<PixelClusterParameterEstimator> 
PixelCPETemplateRecoESProducer::produce(const TkPixelCPERecord & iRecord){ 

  ESHandle<MagneticField> magfield;
  iRecord.getRecord<IdealMagneticFieldRecord>().get(magfield );

  edm::ESHandle<TrackerGeometry> pDD;
  iRecord.getRecord<TrackerDigiGeometryRecord>().get( pDD );

  edm::ESHandle<SiPixelLorentzAngle> lorentzAngle;
  if(DoLorentz_) {
    //  LA from alignment 
    iRecord.getRecord<SiPixelLorentzAngleRcd>().get("laFromAlignment",lorentzAngle);
  } else { // Normal, deafult LA from calibrations
    iRecord.getRecord<SiPixelLorentzAngleRcd>().get(lorentzAngle);
  }

  ESHandle<SiPixelTemplateDBObject> templateDBobject;
  iRecord.getRecord<SiPixelTemplateDBObjectESProducerRcd>().get(templateDBobject);

  cpe_  = boost::shared_ptr<PixelClusterParameterEstimator>(new PixelCPETemplateReco(pset_,magfield.product(),lorentzAngle.product(),templateDBobject.product() ));
  return cpe_;
}



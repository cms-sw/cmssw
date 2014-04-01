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

// new record 
#include "CondFormats/DataRecord/interface/SiPixelGenErrorDBObjectRcd.h"


#include <string>
#include <memory>

using namespace edm;

#define NEW

PixelCPEGenericESProducer::PixelCPEGenericESProducer(const edm::ParameterSet & p) 
{
  std::string myname = p.getParameter<std::string>("ComponentName");
  // Use LA-width from DB. If both (upper and this) are false LA-width is calcuated from LA-offset
  //useLAWidthFromDB_ = p.getParameter<bool>("useLAWidthFromDB");
  useLAWidthFromDB_ = p.existsAs<bool>("useLAWidthFromDB")?
    p.getParameter<bool>("useLAWidthFromDB"):false;
  // Use Alignment LA-offset 
  //useLAAlignmentOffsets_ = p.getParameter<bool>("useLAAlignmentOffsets");
  useLAAlignmentOffsets_ = p.existsAs<bool>("useLAAlignmentOffsets")?
    p.getParameter<bool>("useLAAlignmentOffsets"):false;

  pset_ = p;
  setWhatProduced(this,myname);

  //std::cout<<" ESProducer "<<myname<<" "<<useLAWidthFromDB_<<" "<<useLAAlignmentOffsets_<<std::endl; //dk

}

PixelCPEGenericESProducer::~PixelCPEGenericESProducer() {}

boost::shared_ptr<PixelClusterParameterEstimator>
PixelCPEGenericESProducer::produce(const TkPixelCPERecord & iRecord){ 

  ESHandle<MagneticField> magfield;
  iRecord.getRecord<IdealMagneticFieldRecord>().get( magfield );

  edm::ESHandle<TrackerGeometry> pDD;
  iRecord.getRecord<TrackerDigiGeometryRecord>().get( pDD );

  // Lorant angle for offsets
  ESHandle<SiPixelLorentzAngle> lorentzAngle;
  if(useLAAlignmentOffsets_) // LA offsets from alignment 
    iRecord.getRecord<SiPixelLorentzAngleRcd>().get("fromAlignment",lorentzAngle );
  else // standard LA, from calibration, label=""
    iRecord.getRecord<SiPixelLorentzAngleRcd>().get(lorentzAngle );

  // add the new la width object
  ESHandle<SiPixelLorentzAngle> lorentzAngleWidth;
  const SiPixelLorentzAngle * lorentzAngleWidthProduct = 0;
  if(useLAWidthFromDB_) { // use the width LA
    iRecord.getRecord<SiPixelLorentzAngleRcd>().get("forWidth",lorentzAngleWidth );
    lorentzAngleWidthProduct = lorentzAngleWidth.product();
  } else { lorentzAngleWidthProduct = NULL;} // do not use it
  //std::cout<<" la width "<<lorentzAngleWidthProduct<<std::endl; //dk

  // do we still need this?	
  ESHandle<SiPixelCPEGenericErrorParm> genErrorParm;
  iRecord.getRecord<SiPixelCPEGenericErrorParmRcd>().get(genErrorParm);

  const SiPixelGenErrorDBObject * genErrorDBObjectProduct = 0;
  const bool useGenErrors = false;
  if(useGenErrors) { // new genError object
    ESHandle<SiPixelGenErrorDBObject> genErrorDBObject;
    //iRecord.getRecord<SiPixelGenErrorDBObjectRcd>().get(genErrorDBObject); //this probably needs new TKPixelCPERecord.h
    genErrorDBObjectProduct = genErrorDBObject.product();
  }

  // errors come from this, replace by a lighter object
  ESHandle<SiPixelTemplateDBObject> templateDBobject;
  iRecord.getRecord<SiPixelTemplateDBObjectESProducerRcd>().get(templateDBobject);

#ifdef NEW
  cpe_  = boost::shared_ptr<PixelClusterParameterEstimator>(new PixelCPEGeneric(
	  pset_,magfield.product(),*pDD.product(),lorentzAngle.product(),genErrorDBObjectProduct,
          templateDBobject.product(),lorentzAngleWidthProduct) );
#else
  cpe_  = boost::shared_ptr<PixelClusterParameterEstimator>(new PixelCPEGeneric(
	  pset_,magfield.product(),*pDD.product(),lorentzAngle.product(),0,
          templateDBobject.product(),lorentzAngleWidthProduct) );
#endif

  return cpe_;
}



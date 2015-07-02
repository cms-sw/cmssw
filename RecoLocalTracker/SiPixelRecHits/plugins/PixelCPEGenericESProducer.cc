#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEGenericESProducer.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEGeneric.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

// new record 
#include "CondFormats/DataRecord/interface/SiPixelGenErrorDBObjectRcd.h"


#include <string>
#include <memory>

#define NEW_CPEERROR // must be constistent with base.cc, generic cc/h and genericProducer.cc 

using namespace edm;

PixelCPEGenericESProducer::PixelCPEGenericESProducer(const edm::ParameterSet & p) 
{
  std::string myname = p.getParameter<std::string>("ComponentName");
  // Use LA-width from DB. If both (upper and this) are false LA-width is calcuated from LA-offset
  useLAWidthFromDB_ = p.existsAs<bool>("useLAWidthFromDB")?
    p.getParameter<bool>("useLAWidthFromDB"):false;
  // Use Alignment LA-offset 
  useLAAlignmentOffsets_ = p.existsAs<bool>("useLAAlignmentOffsets")?
    p.getParameter<bool>("useLAAlignmentOffsets"):false;
  magname_ = p.existsAs<edm::ESInputTag>("MagneticFieldRecord")?
    p.getParameter<edm::ESInputTag>("MagneticFieldRecord"):edm::ESInputTag("");
  UseErrorsFromTemplates_    = p.getParameter<bool>("UseErrorsFromTemplates");


  pset_ = p;
  setWhatProduced(this,myname);

  //std::cout<<" ESProducer "<<myname<<" "<<useLAWidthFromDB_<<" "<<useLAAlignmentOffsets_<<" "
  //	   <<UseErrorsFromTemplates_<<std::endl; //dk

}

PixelCPEGenericESProducer::~PixelCPEGenericESProducer() {}

boost::shared_ptr<PixelClusterParameterEstimator>
PixelCPEGenericESProducer::produce(const TkPixelCPERecord & iRecord){ 

  ESHandle<MagneticField> magfield;
  iRecord.getRecord<IdealMagneticFieldRecord>().get( magname_, magfield );

  edm::ESHandle<TrackerGeometry> pDD;
  iRecord.getRecord<TrackerDigiGeometryRecord>().get( pDD );

  edm::ESHandle<TrackerTopology> hTT;
  iRecord.getRecord<TrackerDigiGeometryRecord>().getRecord<TrackerTopologyRcd>().get(hTT);

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

  const SiPixelGenErrorDBObject * genErrorDBObjectProduct = 0;

#ifdef NEW_CPEERROR
  // Errors take only from new GenError
  ESHandle<SiPixelGenErrorDBObject> genErrorDBObject;
  if(UseErrorsFromTemplates_) {  // do only when generrors are needed
    iRecord.getRecord<SiPixelGenErrorDBObjectRcd>().get(genErrorDBObject); 
    genErrorDBObjectProduct = genErrorDBObject.product();
    //} else {
    //std::cout<<" pass an empty GenError pointer"<<std::endl;
  }
  cpe_  = boost::shared_ptr<PixelClusterParameterEstimator>
    (new PixelCPEGeneric(pset_,magfield.product(),*pDD.product(),
			 *hTT.product(),lorentzAngle.product(),
			 genErrorDBObjectProduct,lorentzAngleWidthProduct) );

#else  // old full templates, not used anymore  
  // Errors can be used from tempaltes or from GenError, for testing only
  const bool useNewSimplerErrors = false;
  if(useNewSimplerErrors) { // new genError object
    ESHandle<SiPixelGenErrorDBObject> genErrorDBObject;
    iRecord.getRecord<SiPixelGenErrorDBObjectRcd>().get(genErrorDBObject); //needs new TKPixelCPERecord.h
    genErrorDBObjectProduct = genErrorDBObject.product();
  }

  // errors come from templates
  ESHandle<SiPixelTemplateDBObject> templateDBobject;
  iRecord.getRecord<SiPixelTemplateDBObjectESProducerRcd>().get(templateDBobject);

  cpe_  = boost::shared_ptr<PixelClusterParameterEstimator>(new PixelCPEGeneric(
										pset_,magfield.product(),*pDD.product(),*hTT.product(),lorentzAngle.product(),genErrorDBObjectProduct,
										templateDBobject.product(),lorentzAngleWidthProduct) );
#endif

  return cpe_;
}



// -*- C++ -*-
//
// Package:    SiPixelGenErrorDBObjectESProducer
// Class:      SiPixelGenErrorDBObjectESProducer
// 
/**\class SiPixelGenErrorDBObjectESProducer SiPixelGenErrorDBObjectESProducer.cc CalibTracker/SiPixelESProducers/plugin/SiPixelGenErrorDBObjectESProducer.cc

 Description: ESProducer for magnetic-field-dependent local reco GenErrors

 Implementation: Used inside the RecoLocalTracker/Records/TkPixelRecord to select the correct db for given magnetic field
*/
//
// Original Author:  D.Fehling
//         Created:  Tue Sep 29 14:49:31 CET 2009
//
//

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/do_nothing_deleter.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelGenErrorDBObject.h"
#include "CalibTracker/Records/interface/SiPixelGenErrorDBObjectESProducerRcd.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include <memory>


using namespace edm;

class SiPixelGenErrorDBObjectESProducer : public edm::ESProducer  {

public:

  SiPixelGenErrorDBObjectESProducer(const edm::ParameterSet& iConfig);
  ~SiPixelGenErrorDBObjectESProducer() override;
  std::shared_ptr<const SiPixelGenErrorDBObject> produce(const SiPixelGenErrorDBObjectESProducerRcd &);
};


SiPixelGenErrorDBObjectESProducer::SiPixelGenErrorDBObjectESProducer(const edm::ParameterSet& iConfig) {
	setWhatProduced(this);
}


SiPixelGenErrorDBObjectESProducer::~SiPixelGenErrorDBObjectESProducer(){
}




std::shared_ptr<const SiPixelGenErrorDBObject> SiPixelGenErrorDBObjectESProducer::produce(const SiPixelGenErrorDBObjectESProducerRcd & iRecord) {
	
	ESHandle<MagneticField> magfield;
	iRecord.getRecord<IdealMagneticFieldRecord>().get(magfield);

	GlobalPoint center(0.0, 0.0, 0.0);
	float theMagField = magfield.product()->inTesla(center).mag();

	std::string label = "";
	
	if(     theMagField>=-0.1 && theMagField<1.0 ) label = "0T";
	else if(theMagField>=1.0  && theMagField<2.5 ) label = "2T";
	else if(theMagField>=2.5  && theMagField<3.25) label = "3T";
	else if(theMagField>=3.25 && theMagField<3.65) label = "35T";
	else if(theMagField>=3.9  && theMagField<4.1 ) label = "4T";
	else {
		//label = "3.8T";
		if(theMagField>=4.1 || theMagField<-0.1) edm::LogWarning("UnexpectedMagneticFieldUsingDefaultPixelGenError") << "Magnetic field is " << theMagField;
	}
	ESHandle<SiPixelGenErrorDBObject> dbobject;
	iRecord.getRecord<SiPixelGenErrorDBObjectRcd>().get(label,dbobject);

	if(std::fabs(theMagField-dbobject->sVector()[22])>0.1)
		edm::LogWarning("UnexpectedMagneticFieldUsingNonIdealPixelGenError") << "Magnetic field is " << theMagField << " GenError Magnetic field is " << dbobject->sVector()[22];
	
	return std::shared_ptr<const SiPixelGenErrorDBObject>(&(*dbobject), edm::do_nothing_deleter());
}

DEFINE_FWK_EVENTSETUP_MODULE(SiPixelGenErrorDBObjectESProducer);

// -*- C++ -*-
// Package:    SiPixelESProducers
// Class:      SiPixelTemplateDBObjectESProducer
// Original Author:  D.Fehling
//         Created:  Tue Sep 29 14:49:31 CET 2009
//

#include "CalibTracker/SiPixelESProducers/interface/SiPixelTemplateDBObjectESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <boost/shared_ptr.hpp>
#include "boost/mpl/vector.hpp"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "MagneticField/Engine/interface/MagneticField.h"

using namespace edm;

SiPixelTemplateDBObjectESProducer::SiPixelTemplateDBObjectESProducer(const edm::ParameterSet& iConfig) {
	setWhatProduced(this);
}


SiPixelTemplateDBObjectESProducer::~SiPixelTemplateDBObjectESProducer(){
}




boost::shared_ptr<SiPixelTemplateDBObject> SiPixelTemplateDBObjectESProducer::produce(const SiPixelTemplateDBObjectESProducerRcd & iRecord) {
	
	ESHandle<MagneticField> magfield;
	iRecord.getRecord<IdealMagneticFieldRecord>().get(magfield);

	GlobalPoint center(0.0, 0.0, 0.0);
	float theMagField = magfield.product()->inTesla(center).mag();

	if(theMagField==0) {
		ESHandle<SiPixelTemplateDBObject> SiPixelTemplateDBObject0T;
		iRecord.getRecord<SiPixelTemplateDBObject0TRcd>().get(SiPixelTemplateDBObject0T);
		boost::shared_ptr<SiPixelTemplateDBObject> dbptr(new SiPixelTemplateDBObject(*(SiPixelTemplateDBObject0T)));
		return dbptr;
	}
	else if(theMagField>3.9 && theMagField<4.1) {
		ESHandle<SiPixelTemplateDBObject> SiPixelTemplateDBObject4T;
		iRecord.getRecord<SiPixelTemplateDBObject4TRcd>().get(SiPixelTemplateDBObject4T);
		boost::shared_ptr<SiPixelTemplateDBObject> dbptr(new SiPixelTemplateDBObject(*(SiPixelTemplateDBObject4T)));
		return dbptr;
	}
	else {
		if(theMagField<3.7 || theMagField>=4.1) edm::LogWarning("UnexpectedMagneticFieldUsingDefaultPixelTemplate") << "Magnetic field is " << theMagField;
		ESHandle<SiPixelTemplateDBObject> SiPixelTemplateDBObject38T;
		iRecord.getRecord<SiPixelTemplateDBObject38TRcd>().get(SiPixelTemplateDBObject38T);
		boost::shared_ptr<SiPixelTemplateDBObject> dbptr(new SiPixelTemplateDBObject(*(SiPixelTemplateDBObject38T)));
		return dbptr;
	}
}

DEFINE_FWK_EVENTSETUP_MODULE(SiPixelTemplateDBObjectESProducer);

// -*- C++ -*-
// Package:    SiPixelESProducers
// Class:      SiPixel2DTemplateDBObjectESProducer
// Original Author:  D.Fehling
//         Created:  Tue Sep 29 14:49:31 CET 2009
//

#include "CalibTracker/SiPixelESProducers/interface/SiPixel2DTemplateDBObjectESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/do_nothing_deleter.h"

#include <memory>
#include "boost/mpl/vector.hpp"

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "MagneticField/Engine/interface/MagneticField.h"

using namespace edm;

SiPixel2DTemplateDBObjectESProducer::SiPixel2DTemplateDBObjectESProducer(const edm::ParameterSet& iConfig) {
	setWhatProduced(this);
}


SiPixel2DTemplateDBObjectESProducer::~SiPixel2DTemplateDBObjectESProducer(){
}




std::shared_ptr<SiPixel2DTemplateDBObject> SiPixel2DTemplateDBObjectESProducer::produce(const SiPixel2DTemplateDBObjectESProducerRcd & iRecord) {
	
	ESHandle<MagneticField> magfield;
	iRecord.getRecord<IdealMagneticFieldRecord>().get(magfield);

	GlobalPoint center(0.0, 0.0, 0.0);
	float theMagField = magfield.product()->inTesla(center).mag();

    std::string label = "numerator";   // The correct default
	//  std::string label = "denominator"; // Outdated. Used for MC in older GT's
	//  std::string label = "";      // Outdated. Old default 

	if(     theMagField>=-0.1 && theMagField<1.0 ) label = "0T";
	else if(theMagField>=1.0  && theMagField<2.5 ) label = "2T";
	else if(theMagField>=2.5  && theMagField<3.25) label = "3T";
	else if(theMagField>=3.25 && theMagField<3.65) label = "35T";
	else if(theMagField>=3.9  && theMagField<4.1 ) label = "4T";
	else {
		//label = "3.8T";
		if(theMagField>=4.1 || theMagField<-0.1) edm::LogWarning("UnexpectedMagneticFieldUsingDefaultPixel2DTemplate") << "Magnetic field is " << theMagField;
	}
	ESHandle<SiPixel2DTemplateDBObject> dbobject;
	iRecord.getRecord<SiPixel2DTemplateDBObjectRcd>().get(label,dbobject);

	if(std::fabs(theMagField-dbobject->sVector()[22])>0.1)
		edm::LogWarning("UnexpectedMagneticFieldUsingNonIdealPixel2DTemplate") << "Magnetic field is " << theMagField << " Template Magnetic field is " << dbobject->sVector()[22];
	
	return std::shared_ptr<SiPixel2DTemplateDBObject>(const_cast<SiPixel2DTemplateDBObject*>(&(*dbobject)), edm::do_nothing_deleter());
}

DEFINE_FWK_EVENTSETUP_MODULE(SiPixel2DTemplateDBObjectESProducer);

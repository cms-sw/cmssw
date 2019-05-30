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

private:
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magfieldToken_;
  edm::ESGetToken<SiPixelGenErrorDBObject, SiPixelGenErrorDBObjectRcd> dbTokenDefault_;
  edm::ESGetToken<SiPixelGenErrorDBObject, SiPixelGenErrorDBObjectRcd> dbToken0T_;
  edm::ESGetToken<SiPixelGenErrorDBObject, SiPixelGenErrorDBObjectRcd> dbToken2T_;
  edm::ESGetToken<SiPixelGenErrorDBObject, SiPixelGenErrorDBObjectRcd> dbToken3T_;
  edm::ESGetToken<SiPixelGenErrorDBObject, SiPixelGenErrorDBObjectRcd> dbToken35T_;
  edm::ESGetToken<SiPixelGenErrorDBObject, SiPixelGenErrorDBObjectRcd> dbToken4T_;
};


SiPixelGenErrorDBObjectESProducer::SiPixelGenErrorDBObjectESProducer(const edm::ParameterSet& iConfig) {
  setWhatProduced(this)
    .setConsumes(magfieldToken_)
    .setConsumes(dbTokenDefault_, edm::ESInputTag{"", ""})
    .setConsumes(dbToken0T_, edm::ESInputTag{"", "0T"})
    .setConsumes(dbToken2T_, edm::ESInputTag{"", "2T"})
    .setConsumes(dbToken3T_, edm::ESInputTag{"", "3T"})
    .setConsumes(dbToken35T_, edm::ESInputTag{"", "35T"})
    .setConsumes(dbToken4T_, edm::ESInputTag{"", "4T"});
;
}


SiPixelGenErrorDBObjectESProducer::~SiPixelGenErrorDBObjectESProducer(){
}




std::shared_ptr<const SiPixelGenErrorDBObject> SiPixelGenErrorDBObjectESProducer::produce(const SiPixelGenErrorDBObjectESProducerRcd & iRecord) {
	const auto& magfield = iRecord.get(magfieldToken_);

	GlobalPoint center(0.0, 0.0, 0.0);
	float theMagField = magfield.inTesla(center).mag();

        const auto *tokenPtr = &dbTokenDefault_;

	if(     theMagField>=-0.1 && theMagField<1.0 ) tokenPtr = &dbToken0T_;
	else if(theMagField>=1.0  && theMagField<2.5 ) tokenPtr = &dbToken2T_;
	else if(theMagField>=2.5  && theMagField<3.25) tokenPtr = &dbToken3T_;
	else if(theMagField>=3.25 && theMagField<3.65) tokenPtr = &dbToken35T_;
	else if(theMagField>=3.9  && theMagField<4.1 ) tokenPtr = &dbToken4T_;
	else {
		//label = "3.8T";
		if(theMagField>=4.1 || theMagField<-0.1) edm::LogWarning("UnexpectedMagneticFieldUsingDefaultPixelGenError") << "Magnetic field is " << theMagField;
	}
	const auto& dbobject = iRecord.get(*tokenPtr);

	if(std::fabs(theMagField-dbobject.sVector()[22])>0.1)
		edm::LogWarning("UnexpectedMagneticFieldUsingNonIdealPixelGenError") << "Magnetic field is " << theMagField << " GenError Magnetic field is " << dbobject.sVector()[22];
	
	return std::shared_ptr<const SiPixelGenErrorDBObject>(&dbobject, edm::do_nothing_deleter());
}

DEFINE_FWK_EVENTSETUP_MODULE(SiPixelGenErrorDBObjectESProducer);

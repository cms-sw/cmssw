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
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESProductTag.h"
#include "FWCore/Utilities/interface/do_nothing_deleter.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelGenErrorDBObject.h"
#include "CalibTracker/Records/interface/SiPixelGenErrorDBObjectESProducerRcd.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include <memory>

using namespace edm;

class SiPixelGenErrorDBObjectESProducer : public edm::ESProducer {
public:
  SiPixelGenErrorDBObjectESProducer(const edm::ParameterSet& iConfig);
  std::shared_ptr<const SiPixelGenErrorDBObject> produce(const SiPixelGenErrorDBObjectESProducerRcd&);

private:
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magfieldToken_;
  edm::ESGetToken<SiPixelGenErrorDBObject, SiPixelGenErrorDBObjectRcd> genErrorToken_;
};

SiPixelGenErrorDBObjectESProducer::SiPixelGenErrorDBObjectESProducer(const edm::ParameterSet& iConfig) {
  setWhatProduced(this)
      .setMayConsume(
          genErrorToken_,
          [](const auto& get, edm::ESTransientHandle<MagneticField> iMagfield) {
            const GlobalPoint center(0.0, 0.0, 0.0);
            const float theMagField = iMagfield->inTesla(center).mag();
            if (theMagField >= -0.1 && theMagField < 1.0)
              return get("", "0T");
            else if (theMagField >= 1.0 && theMagField < 2.5)
              return get("", "2T");
            else if (theMagField >= 2.5 && theMagField < 3.25)
              return get("", "3T");
            else if (theMagField >= 3.25 && theMagField < 3.65)
              return get("", "35T");
            else if (theMagField >= 3.9 && theMagField < 4.1)
              return get("", "4T");
            else {
              if (theMagField >= 4.1 || theMagField < -0.1)
                edm::LogWarning("UnexpectedMagneticFieldUsingDefaultPixelGenError")
                    << "Magnetic field is " << theMagField;
              //return get("", "3.8T");
              return get("", "");
            }
          },
          edm::ESProductTag<MagneticField, IdealMagneticFieldRecord>("", ""))
      .setConsumes(magfieldToken_);
}

std::shared_ptr<const SiPixelGenErrorDBObject> SiPixelGenErrorDBObjectESProducer::produce(
    const SiPixelGenErrorDBObjectESProducerRcd& iRecord) {
  const GlobalPoint center(0.0, 0.0, 0.0);
  const float theMagField = iRecord.get(magfieldToken_).inTesla(center).mag();

  const auto& dbobject = iRecord.get(genErrorToken_);

  if (std::fabs(theMagField - dbobject.sVector()[22]) > 0.1)
    edm::LogWarning("UnexpectedMagneticFieldUsingNonIdealPixelGenError")
        << "Magnetic field is " << theMagField << " GenError Magnetic field is " << dbobject.sVector()[22];

  return std::shared_ptr<const SiPixelGenErrorDBObject>(&dbobject, edm::do_nothing_deleter());
}

DEFINE_FWK_EVENTSETUP_MODULE(SiPixelGenErrorDBObjectESProducer);

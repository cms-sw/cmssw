// -*- C++ -*-
//
// Package:    SiPixel2DTemplateDBObjectESProducer
// Class:      SiPixel2DTemplateDBObjectESProducer
//
/**\class SiPixel2DTemplateDBObjectESProducer SiPixel2DTemplateDBObjectESProducer.cc CalibTracker/SiPixelESProducers/plugin/SiPixel2DTemplateDBObjectESProducer.cc

 Description: ESProducer for magnetic-field-dependent local reco templates

 Implementation: Used inside the RecoLocalTracker/Records/TkPixelRecord to select the correct db for given magnetic field
*/
//
// Original Author:  D.Fehling
//         Created:  Tue Sep 29 14:49:31 CET 2009
//
//

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/do_nothing_deleter.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "CondFormats/SiPixelObjects/interface/SiPixel2DTemplateDBObject.h"
#include "CalibTracker/Records/interface/SiPixel2DTemplateDBObjectESProducerRcd.h"

#include <memory>

using namespace edm;

class SiPixel2DTemplateDBObjectESProducer : public edm::ESProducer {
public:
  SiPixel2DTemplateDBObjectESProducer(const edm::ParameterSet& iConfig);
  ~SiPixel2DTemplateDBObjectESProducer() override;
  std::shared_ptr<const SiPixel2DTemplateDBObject> produce(const SiPixel2DTemplateDBObjectESProducerRcd&);

private:
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magfieldToken_;
  edm::ESGetToken<SiPixel2DTemplateDBObject, SiPixel2DTemplateDBObjectRcd> dbToken_;
};

SiPixel2DTemplateDBObjectESProducer::SiPixel2DTemplateDBObjectESProducer(const edm::ParameterSet& iConfig) {
  auto cc = setWhatProduced(this);
  magfieldToken_ = cc.consumes();
  dbToken_ = cc.consumes(edm::ESInputTag{"", "numerator"});  // The correct default
}

SiPixel2DTemplateDBObjectESProducer::~SiPixel2DTemplateDBObjectESProducer() {}

std::shared_ptr<const SiPixel2DTemplateDBObject> SiPixel2DTemplateDBObjectESProducer::produce(
    const SiPixel2DTemplateDBObjectESProducerRcd& iRecord) {
  const auto& magfield = iRecord.get(magfieldToken_);

  GlobalPoint center(0.0, 0.0, 0.0);
  float theMagField = magfield.inTesla(center).mag();

  if (theMagField >= 4.1 || theMagField < -0.1)
    edm::LogWarning("UnexpectedMagneticFieldUsingDefaultPixel2DTemplate") << "Magnetic field is " << theMagField;

  const auto& dbobject = iRecord.get(dbToken_);

  if ((theMagField > 0.1) && (std::fabs(theMagField - dbobject.sVector()[22]) > 0.1))
    //2D templates not actually used at 0T, so don't print warning
    edm::LogWarning("UnexpectedMagneticFieldUsingNonIdealPixel2DTemplate")
        << "Magnetic field is " << theMagField << " Template Magnetic field is " << dbobject.sVector()[22];

  return std::shared_ptr<const SiPixel2DTemplateDBObject>(&dbobject, edm::do_nothing_deleter());
}

DEFINE_FWK_EVENTSETUP_MODULE(SiPixel2DTemplateDBObjectESProducer);

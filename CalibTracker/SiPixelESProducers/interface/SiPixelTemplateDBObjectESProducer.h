#ifndef CalibTracker_SiPixelESProducers_SiPixelTemplateDBObjectESProducer_h
#define CalibTracker_SiPixelESProducers_SiPixelTemplateDBObjectESProducer_h
// -*- C++ -*-
//
// Package:    SiPixelTemplateDBObjectESProducer
// Class:      SiPixelTemplateDBObjectESProducer
// 
/**\class SiPixelTemplateDBObjectESProducer SiPixelTemplateDBObjectESProducer.cc CalibTracker/SiPixelESProducers/plugin/SiPixelTemplateDBObjectESProducer.cc

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
#include "CondFormats/SiPixelObjects/interface/SiPixelTemplateDBObject.h"
#include "CalibTracker/Records/interface/SiPixelTemplateDBObjectESProducerRcd.h"

class SiPixelTemplateDBObjectESProducer : public edm::ESProducer  {

public:

  SiPixelTemplateDBObjectESProducer(const edm::ParameterSet& iConfig);
  ~SiPixelTemplateDBObjectESProducer() override;
  std::shared_ptr<const SiPixelTemplateDBObject> produce(const SiPixelTemplateDBObjectESProducerRcd &);
};
#endif

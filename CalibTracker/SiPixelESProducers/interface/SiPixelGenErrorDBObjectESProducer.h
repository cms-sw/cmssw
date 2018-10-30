#ifndef CalibTracker_SiPixelESProducers_SiPixelGenErrorDBObjectESProducer_h
#define CalibTracker_SiPixelESProducers_SiPixelGenErrorDBObjectESProducer_h
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
#include "CondFormats/SiPixelObjects/interface/SiPixelGenErrorDBObject.h"
#include "CalibTracker/Records/interface/SiPixelGenErrorDBObjectESProducerRcd.h"

class SiPixelGenErrorDBObjectESProducer : public edm::ESProducer  {

public:

  SiPixelGenErrorDBObjectESProducer(const edm::ParameterSet& iConfig);
  ~SiPixelGenErrorDBObjectESProducer() override;
  std::shared_ptr<const SiPixelGenErrorDBObject> produce(const SiPixelGenErrorDBObjectESProducerRcd &);
};
#endif

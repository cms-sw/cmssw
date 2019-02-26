#ifndef CalibTracker_SiPixelESProducers_SiPixel2DTemplateDBObjectESProducer_h
#define CalibTracker_SiPixelESProducers_SiPixel2DTemplateDBObjectESProducer_h
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
#include "CondFormats/SiPixelObjects/interface/SiPixel2DTemplateDBObject.h"
#include "CalibTracker/Records/interface/SiPixel2DTemplateDBObjectESProducerRcd.h"

class SiPixel2DTemplateDBObjectESProducer : public edm::ESProducer  {

public:

	SiPixel2DTemplateDBObjectESProducer(const edm::ParameterSet& iConfig);
        ~SiPixel2DTemplateDBObjectESProducer() override;
	std::shared_ptr<const SiPixel2DTemplateDBObject> produce(const SiPixel2DTemplateDBObjectESProducerRcd &);
 };
#endif

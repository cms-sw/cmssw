// -*- C++ -*-
//
// Package:    ParametrizedMagneticFieldProducer
// Class:      ParametrizedMagneticFieldProducer
// 
/**\class ParametrizedMagneticFieldProducer ParametrizedMagneticFieldProducer.h MagneticField/ParametrizedEngine/interface/ParametrizedMagneticFieldProducer.h

 Description: Producer for the Parametrized Magnetic Field

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Massimiliano Chiorboli
//         Created:  Tue Jun 28 14:35:40 CEST 2007
// $Id$
//
//


// system include files
#include <memory>

#include "MagneticField/ParametrizedEngine/interface/ParametrizedMagneticFieldProducer.h"
#include "MagneticField/ParametrizedEngine/interface/ParametrizedMagneticField.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/Framework/interface/ModuleFactory.h"

#include <string>
#include <iostream>


using namespace magneticfield;


ParametrizedMagneticFieldProducer::ParametrizedMagneticFieldProducer(
								       const edm::ParameterSet& iConfig
								       ) :
  a_(iConfig.getParameter<double>("paramFieldRadius")),
  l_(iConfig.getParameter<double>("paramFieldLength"))
{
   setWhatProduced(this);
}


ParametrizedMagneticFieldProducer::~ParametrizedMagneticFieldProducer()
{
}


std::auto_ptr<MagneticField>
ParametrizedMagneticFieldProducer::produce(const IdealMagneticFieldRecord& iRecord)
{
   std::auto_ptr<MagneticField> pMagneticField(new ParametrizedMagneticField(a_, l_));
   return pMagneticField ;
}

DEFINE_FWK_EVENTSETUP_MODULE(ParametrizedMagneticFieldProducer);

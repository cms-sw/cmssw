#ifndef ParametrizedMagneticFieldProducer_h
#define ParametrizedMagneticFieldProducer_h

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
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"



//
// class decleration
//

namespace magneticfield {
  class ParametrizedMagneticFieldProducer : public edm::ESProducer
  {
  public:
    ParametrizedMagneticFieldProducer(const edm::ParameterSet&);
    ~ParametrizedMagneticFieldProducer();
    
    std::auto_ptr<MagneticField> produce(const IdealMagneticFieldRecord&);
    
  private:
    float a_;
    float l_;
  };
}
#endif

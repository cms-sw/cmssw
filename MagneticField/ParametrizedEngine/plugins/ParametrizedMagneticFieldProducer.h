#ifndef ParametrizedMagneticFieldProducer_h
#define ParametrizedMagneticFieldProducer_h

/** \class ParametrizedMagneticFieldProducer
 *
 *   Description: Producer for the Parametrized Magnetic Field
 *
 *  \author Massimiliano Chiorboli, updated NA 03/08
 */

#include "FWCore/Framework/interface/ESProducer.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class IdealMagneticFieldRecord;

namespace magneticfield {
  class ParametrizedMagneticFieldProducer : public edm::ESProducer
  {
  public:
    ParametrizedMagneticFieldProducer(const edm::ParameterSet&);
    ~ParametrizedMagneticFieldProducer() override;
    
    std::unique_ptr<MagneticField> produce(const IdealMagneticFieldRecord&);
    edm::ParameterSet pset;
  };
}

#endif

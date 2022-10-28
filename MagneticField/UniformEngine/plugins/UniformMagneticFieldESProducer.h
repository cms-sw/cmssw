#ifndef UniformMagneticFieldESProducer_h
#define UniformMagneticFieldESProducer_h

/** \class UniformMagneticFieldESProducer
 *
 *  Producer for the UniformMagneticField.
 *
 *  \author N. Amapane - CERN
 */

#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class IdealMagneticFieldRecord;

namespace magneticfield {
  class UniformMagneticFieldESProducer : public edm::ESProducer {
  public:
    UniformMagneticFieldESProducer(const edm::ParameterSet& pset);
    // forbid copy ctor and assignment op.
    UniformMagneticFieldESProducer(const UniformMagneticFieldESProducer&) = delete;
    const UniformMagneticFieldESProducer& operator=(const UniformMagneticFieldESProducer&) = delete;

    std::unique_ptr<MagneticField> produce(const IdealMagneticFieldRecord&);

  private:
    float value;
  };
}  // namespace magneticfield

#endif

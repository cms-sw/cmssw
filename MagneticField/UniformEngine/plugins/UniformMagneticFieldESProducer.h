#ifndef UniformMagneticFieldESProducer_h
#define UniformMagneticFieldESProducer_h

/** \class UniformMagneticFieldESProducer
 *
 *  Producer for the UniformMagneticField.
 *
 *  $Date: 2008/11/14 00:19:33 $
 *  $Revision: 1.1 $
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
  
    std::auto_ptr<MagneticField> produce(const IdealMagneticFieldRecord &);

  private:
    // forbid copy ctor and assignment op.
    UniformMagneticFieldESProducer(const UniformMagneticFieldESProducer&);
    const UniformMagneticFieldESProducer& operator=(const UniformMagneticFieldESProducer&);

    float value;
  };
}


#endif

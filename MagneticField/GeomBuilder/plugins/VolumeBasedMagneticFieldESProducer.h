#ifndef VolumeBasedMagneticFieldESProducer_h
#define VolumeBasedMagneticFieldESProducer_h

/** \class VolumeBasedMagneticFieldESProducer
 *
 *  Producer for the VolumeBasedMagneticField.
 *
 */

#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>
#include <vector>

namespace magneticfield {
  class VolumeBasedMagneticFieldESProducer : public edm::ESProducer {
  public:
    VolumeBasedMagneticFieldESProducer(const edm::ParameterSet& iConfig);
  
    std::unique_ptr<MagneticField> produce(const IdealMagneticFieldRecord & iRecord);

  private:
    // forbid copy ctor and assignment op.
    VolumeBasedMagneticFieldESProducer(const VolumeBasedMagneticFieldESProducer&) = delete;
    const VolumeBasedMagneticFieldESProducer& operator=(const VolumeBasedMagneticFieldESProducer&) = delete;

    edm::ParameterSet pset;
  };
}


#endif

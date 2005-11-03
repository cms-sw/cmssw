#ifndef VolumeBasedMagneticFieldESProducer_H
#define VolumeBasedMagneticFieldESProducer_H

// user include files
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <string>
#include <iostream>

namespace magneticfield {
class VolumeBasedMagneticFieldESProducer : 
    public edm::ESProducer
{
public:
  VolumeBasedMagneticFieldESProducer(const edm::ParameterSet&);
  
  std::auto_ptr<MagneticField> produce(const IdealMagneticFieldRecord &);

protected:
private:
  VolumeBasedMagneticFieldESProducer(const VolumeBasedMagneticFieldESProducer&);
  const VolumeBasedMagneticFieldESProducer& operator=(const VolumeBasedMagneticFieldESProducer&);
      // ----------member data ---------------------------
};
}


#endif

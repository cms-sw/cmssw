#ifndef VolumeBasedMagneticFieldESProducer_h
#define VolumeBasedMagneticFieldESProducer_h

/** \class VolumeBasedMagneticFieldESProducer
 *
 *  Producer for the VolumeBasedMagneticField.
 *
 *  $Date: 2008/11/14 10:42:41 $
 *  $Revision: 1.1 $
 */

#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace magneticfield {
  class VolumeBasedMagneticFieldESProducer : public edm::ESProducer {
  public:
    VolumeBasedMagneticFieldESProducer(const edm::ParameterSet& iConfig);
  
    std::auto_ptr<MagneticField> produce(const IdealMagneticFieldRecord & iRecord);

  private:
    // forbid copy ctor and assignment op.
    VolumeBasedMagneticFieldESProducer(const VolumeBasedMagneticFieldESProducer&);
    const VolumeBasedMagneticFieldESProducer& operator=(const VolumeBasedMagneticFieldESProducer&);

    edm::ParameterSet pset;
  };
}


#endif

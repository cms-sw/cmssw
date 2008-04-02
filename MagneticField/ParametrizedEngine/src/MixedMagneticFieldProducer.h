#ifndef MixedMagneticFieldProducer_h
#define MixedMagneticFieldProducer_h

/** \class MixedMagneticFieldProducer
 *
 *   Temporary solution for a patchwork of Magnetic Fields.
 *   The result of this producer may be an unphisical field map!!!
 *   Use at your own risk!!!!
 *
 *  $Date: 2008/03/28 16:49:24 $
 *  $Revision: 1.2 $
 *  \author NA 03/08
 */

#include "FWCore/Framework/interface/ESProducer.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class IdealMagneticFieldRecord;

namespace magneticfield {
  class MixedMagneticFieldProducer : public edm::ESProducer
  {
  public:
    MixedMagneticFieldProducer(const edm::ParameterSet&);
    ~MixedMagneticFieldProducer();
    
    std::auto_ptr<MagneticField> produce(const IdealMagneticFieldRecord&);
    edm::ParameterSet pset;
  };
}

#endif

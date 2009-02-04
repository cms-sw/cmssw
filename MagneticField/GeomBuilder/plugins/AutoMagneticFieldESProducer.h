#ifndef AutoMagneticFieldESProducer_h
#define AutoMagneticFieldESProducer_h

/** \class AutoMagneticFieldESProducer
 *
 *  Produce a magnetic field map corresponding to the current 
 *  recorded in the condidtion DB.
 *
 *  $Date: 2008/04/02 15:51:43 $
 *  $Revision: 1.1 $
 *  \author Nicola Amapane 11/08
 */

#include "FWCore/Framework/interface/ESProducer.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class IdealMagneticFieldRecord;

namespace magneticfield {
  class AutoMagneticFieldESProducer : public edm::ESProducer
  {
  public:
    AutoMagneticFieldESProducer(const edm::ParameterSet&);
    ~AutoMagneticFieldESProducer();
    
    std::auto_ptr<MagneticField> produce(const IdealMagneticFieldRecord&);
    edm::ParameterSet pset;
  };
}

#endif

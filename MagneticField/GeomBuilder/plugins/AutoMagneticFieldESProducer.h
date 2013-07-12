#ifndef AutoMagneticFieldESProducer_h
#define AutoMagneticFieldESProducer_h

/** \class AutoMagneticFieldESProducer
 *
 *  Produce a magnetic field map corresponding to the current 
 *  recorded in the condidtion DB.
 *
 *  $Date: 2008/11/14 10:42:40 $
 *  $Revision: 1.1 $
 *  \author Nicola Amapane 11/08
 */

#include "FWCore/Framework/interface/ESProducer.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <vector>
#include <string>

class IdealMagneticFieldRecord;

namespace magneticfield {
  class AutoMagneticFieldESProducer : public edm::ESProducer
  {
  public:
    AutoMagneticFieldESProducer(const edm::ParameterSet&);
    ~AutoMagneticFieldESProducer();
    
    std::auto_ptr<MagneticField> produce(const IdealMagneticFieldRecord&);
    edm::ParameterSet pset;
  private:
    std::string closerModel(float current);
    std::vector<int> nominalCurrents;
    std::vector<std::string> maps;
  };
}

#endif

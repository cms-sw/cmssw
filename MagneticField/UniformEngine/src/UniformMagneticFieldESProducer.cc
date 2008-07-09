/** \file
 *
 *  $Date: 2006/05/31 13:43:26 $
 *  $Revision: 1.1 $
 *  \author N. Amapane - CERN
 */

#include "MagneticField/UniformEngine/src/UniformMagneticFieldESProducer.h"
#include "MagneticField/UniformEngine/src/UniformMagneticField.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "FWCore/Framework/interface/ModuleFactory.h"

using namespace magneticfield;

UniformMagneticFieldESProducer::UniformMagneticFieldESProducer(const edm::ParameterSet& pset) : value(pset.getParameter<double>("ZFieldInTesla")) {
  setWhatProduced(this);
}


std::auto_ptr<MagneticField> UniformMagneticFieldESProducer::produce(const IdealMagneticFieldRecord & iRecord)
{
  std::auto_ptr<MagneticField> s(new UniformMagneticField(value));
  return s;
}

DEFINE_FWK_EVENTSETUP_MODULE(UniformMagneticFieldESProducer);

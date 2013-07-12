/** \file
 *
 *  $Date: 2008/03/29 11:55:26 $
 *  $Revision: 1.3 $
 *  \author N. Amapane - CERN
 */

#include "MagneticField/UniformEngine/plugins/UniformMagneticFieldESProducer.h"
#include "MagneticField/UniformEngine/src/UniformMagneticField.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "FWCore/Framework/interface/ModuleFactory.h"

using namespace magneticfield;

UniformMagneticFieldESProducer::UniformMagneticFieldESProducer(const edm::ParameterSet& pset) : value(pset.getParameter<double>("ZFieldInTesla")) {
  setWhatProduced(this, pset.getUntrackedParameter<std::string>("label",""));
}


std::auto_ptr<MagneticField> UniformMagneticFieldESProducer::produce(const IdealMagneticFieldRecord & iRecord)
{
  std::auto_ptr<MagneticField> s(new UniformMagneticField(value));
  return s;
}

DEFINE_FWK_EVENTSETUP_MODULE(UniformMagneticFieldESProducer);

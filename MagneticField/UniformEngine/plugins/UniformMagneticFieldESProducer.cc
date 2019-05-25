/** \file
 *
 *  \author N. Amapane - CERN
 */

#include "MagneticField/UniformEngine/plugins/UniformMagneticFieldESProducer.h"
#include "MagneticField/UniformEngine/interface/UniformMagneticField.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "FWCore/Framework/interface/ModuleFactory.h"

using namespace magneticfield;

UniformMagneticFieldESProducer::UniformMagneticFieldESProducer(const edm::ParameterSet& pset)
    : value(pset.getParameter<double>("ZFieldInTesla")) {
  setWhatProduced(this, pset.getUntrackedParameter<std::string>("label", ""));
}

std::unique_ptr<MagneticField> UniformMagneticFieldESProducer::produce(const IdealMagneticFieldRecord& iRecord) {
  return std::make_unique<UniformMagneticField>(value);
}

DEFINE_FWK_EVENTSETUP_MODULE(UniformMagneticFieldESProducer);

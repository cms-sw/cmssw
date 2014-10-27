/** \file
 *
 *  \author Massimiliano Chiorboli, updated NA 03/08
 */

#include "MagneticField/ParametrizedEngine/plugins/ParametrizedMagneticFieldProducer.h"
#include "MagneticField/ParametrizedEngine/interface/ParametrizedMagneticFieldFactory.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>
#include <iostream>

using namespace std;
using namespace edm;
using namespace magneticfield;


ParametrizedMagneticFieldProducer::ParametrizedMagneticFieldProducer(const edm::ParameterSet& iConfig) : pset(iConfig) {
  setWhatProduced(this, pset.getUntrackedParameter<std::string>("label",""));
}


ParametrizedMagneticFieldProducer::~ParametrizedMagneticFieldProducer(){}


std::auto_ptr<MagneticField>
ParametrizedMagneticFieldProducer::produce(const IdealMagneticFieldRecord& iRecord)
{
  string version = pset.getParameter<string>("version");
  ParameterSet parameters = pset.getParameter<ParameterSet>("parameters");

  return ParametrizedMagneticFieldFactory::get(version, parameters);
}

#include "FWCore/Framework/interface/ModuleFactory.h"
DEFINE_FWK_EVENTSETUP_MODULE(ParametrizedMagneticFieldProducer);

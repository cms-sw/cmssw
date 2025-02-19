/** \file
 *
 *  $Date: 2011/04/16 10:20:40 $
 *  $Revision: 1.3 $
 *  \author Massimiliano Chiorboli, updated NA 03/08
 */

#include "MagneticField/ParametrizedEngine/plugins/ParametrizedMagneticFieldProducer.h"
#include "OAEParametrizedMagneticField.h"
#include "OAE85lParametrizedMagneticField.h"
#include "PolyFit2DParametrizedMagneticField.h"
#include "PolyFit3DParametrizedMagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <string>
#include <iostream>

using namespace std;
using namespace edm;
using namespace magneticfield;


ParametrizedMagneticFieldProducer::ParametrizedMagneticFieldProducer(const edm::ParameterSet& iConfig) : pset(iConfig) {
  setWhatProduced(this, pset.getUntrackedParameter<std::string>("label",""));
}


ParametrizedMagneticFieldProducer::~ParametrizedMagneticFieldProducer()
{
}


std::auto_ptr<MagneticField>
ParametrizedMagneticFieldProducer::produce(const IdealMagneticFieldRecord& iRecord)
{
  string version = pset.getParameter<string>("version");
  ParameterSet parameters = pset.getParameter<ParameterSet>("parameters");

  if (version=="OAE_85l_030919") {
    // V. Karimaki's off-axis expansion fitted to v85l TOSCA computation
    std::auto_ptr<MagneticField> result(new OAE85lParametrizedMagneticField(parameters));
    return result;
  } else if (version=="OAE_1103l_071212") {
    // V. Karimaki's off-axis expansion fitted to v1103l TOSCA computation
    std::auto_ptr<MagneticField> result( new OAEParametrizedMagneticField(parameters));
    return result;
  } else if (version=="PolyFit2D") {
    // V. Maroussov polynomial fit to mapping data
    std::auto_ptr<MagneticField> result( new PolyFit2DParametrizedMagneticField(parameters));
    return result;
  } else if (version=="PolyFit3D") {
    // V. Maroussov polynomial fit to mapping data
    std::auto_ptr<MagneticField> result( new PolyFit3DParametrizedMagneticField(parameters));
    return result;
  } else {
    throw cms::Exception("InvalidParameter")<<"Invalid parametrization version " << version;
  }
  return std::auto_ptr<MagneticField>(0); //make compiler happy
}

#include "FWCore/Framework/interface/ModuleFactory.h"
DEFINE_FWK_EVENTSETUP_MODULE(ParametrizedMagneticFieldProducer);

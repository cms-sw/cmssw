/** \file
 *
 *  $Date: 2008/03/28 16:49:25 $
 *  $Revision: 1.2 $
 *  \author Massimiliano Chiorboli, updated NA 03/08
 */

#include "MagneticField/ParametrizedEngine/interface/ParametrizedMagneticFieldProducer.h"
#include "MagneticField/ParametrizedEngine/interface/ParametrizedMagneticField.h"
#include "MagneticField/ParametrizedEngine/src/OAEParametrizedMagneticField.h"
#include "MagneticField/ParametrizedEngine/src/OAE85lParametrizedMagneticField.h"
#include "MagneticField/ParametrizedEngine/src/PolyFit2DParametrizedMagneticField.h"
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
    std::auto_ptr<MagneticField> result(new OAE85lParametrizedMagneticField(parameters));
    return result;
  } else if (version=="OAE_1103l_071212") {
    std::auto_ptr<MagneticField> result( new OAEParametrizedMagneticField(parameters));
    return result;
  } else if (version=="PolyFit2D") {
  // V. Maroussov polynomial fit to mapping data
    std::auto_ptr<MagneticField> result( new PolyFit2DParametrizedMagneticField(parameters));
  //  return result;
    return std::auto_ptr<MagneticField>(0);
  } else if (version=="OAE_85l_030919_t") {
    // Use old field - this should be replaced by OAE_85l_030919!!!
    float a = parameters.getParameter<double>("paramFieldRadius");
    float l = parameters.getParameter<double>("paramFieldLength");
    std::auto_ptr<MagneticField> result(new ParametrizedMagneticField(a,l));
    return result;
  } else {
    throw cms::Exception("InvalidParameter")<<"Invalid parametrization version " << version;
  }
  return std::auto_ptr<MagneticField>(0); //make compiler happy
}

#include "FWCore/Framework/interface/ModuleFactory.h"
DEFINE_FWK_EVENTSETUP_MODULE(ParametrizedMagneticFieldProducer);

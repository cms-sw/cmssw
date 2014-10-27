/** \file
 *
 *  \author N. Amapane - Torino
 */

#include <MagneticField/ParametrizedEngine/interface/ParametrizedMagneticFieldFactory.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include "OAEParametrizedMagneticField.h"
#include "ParabolicParametrizedMagneticField.h"
#include "PolyFit2DParametrizedMagneticField.h"

#include "FWCore/Utilities/interface/Exception.h"

using namespace std;
using namespace edm;

ParametrizedMagneticFieldFactory::ParametrizedMagneticFieldFactory(){}

// Legacy interface, deprecated
std::auto_ptr<MagneticField>
ParametrizedMagneticFieldFactory::get(string version, const ParameterSet& parameters) {

  if (version=="OAE_1103l_071212") {
    // V. Karimaki's off-axis expansion fitted to v1103l TOSCA computation
    std::auto_ptr<MagneticField> result( new OAEParametrizedMagneticField(parameters));
    return result;
  } else if (version=="PolyFit2D") {
    // V. Maroussov polynomial fit to mapping data
    std::auto_ptr<MagneticField> result( new PolyFit2DParametrizedMagneticField(parameters));
    return result;
  } else if (version=="PolyFit3D") {
    // V. Maroussov polynomial fit to mapping data
    throw cms::Exception("InvalidParameter")<<"PolyFit3D is not supported anymore";
  } else if (version=="Parabolic"){
    // FIXME implement configurable parameters to be passed to ctor
//   vector<double> params =  parameters.getParameter<vdouble>("parameters");
//   std::auto_ptr<MagneticField> result( new ParabolicParametrizedMagneticField(params));
    std::auto_ptr<MagneticField> result( new ParabolicParametrizedMagneticField());
    return result;
  }  else {
    throw cms::Exception("InvalidParameter")<<"Invalid parametrization version " << version;
  }
  return std::auto_ptr<MagneticField>(0); //make compiler happy
}


// New interface
std::auto_ptr<MagneticField>
ParametrizedMagneticFieldFactory::get(string version, vector<double> parameters) {

  if (version=="OAE_1103l_071212") {
    // V. Karimaki's off-axis expansion fitted to v1103l TOSCA computation
    if (parameters.size()!=1) throw cms::Exception("InvalidParameter") << "Incorrect parameters (" << parameters.size()<< ")`for " << version ;
    std::auto_ptr<MagneticField> result( new OAEParametrizedMagneticField(parameters[0]));
    return result;
  } else if (version=="PolyFit2D") {
    // V. Maroussov polynomial fit to mapping data
    if (parameters.size()!=1) throw cms::Exception("InvalidParameter") << "Incorrect parameters (" << parameters.size()<< ")`for " << version ;
    std::auto_ptr<MagneticField> result( new PolyFit2DParametrizedMagneticField(parameters[0]));
    return result;
  } else if (version=="PolyFit3D") {
    // V. Maroussov polynomial fit to mapping data
    throw cms::Exception("InvalidParameter")<<"PolyFit3D is not supported anymore";
  } else if (version=="Parabolic"){
    if (parameters.size()!=4) throw cms::Exception("InvalidParameter") << "Incorrect parameters (" << parameters.size()<< ")`for " << version ;
    std::auto_ptr<MagneticField> result( new ParabolicParametrizedMagneticField(parameters));
    return result;
  }  else {
    throw cms::Exception("InvalidParameter")<<"Invalid parametrization version " << version;
  }
  return std::auto_ptr<MagneticField>(0); //make compiler happy
}

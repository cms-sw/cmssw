//   COCOA class implementation file
//Id:  ParameterMgr.cc
//CAT: Model
//
//   History: v1.0  10/11/01   Pedro Arce

#include "Alignment/CocoaModel/interface/ParameterMgr.h"
#include "Alignment/CocoaUtilities/interface/ALIUtils.h"
#include "Alignment/CocoaModel/interface/ALIUnitsTable.h"
#include <CLHEP/Random/RandGauss.h>
#include <CLHEP/Random/Random.h>
#include <cstdlib>
//----------------------------------------------------------------------------

ParameterMgr* ParameterMgr::theInstance = nullptr;

//----------------------------------------------------------------------------
ParameterMgr* ParameterMgr::getInstance() {
  if (!theInstance) {
    theInstance = new ParameterMgr;
  }

  return theInstance;
}

//----------------------------------------------------------------------------
ALIdouble ParameterMgr::getVal(const ALIstring& str, const ALIdouble dimensionFactor) {
  //If there is a '*', the characters after '*' are the unit
  ALIint iast = str.find('*');
  //  ALIdouble vl;
  if (iast != -1) {
    ALIstring valstr = str.substr(0, iast);
    ALIstring unitstr = str.substr(iast + 1, str.length());

    //-    std::cout << iast << "parametermgr " << str << " " << valstr << " " << unitstr << std::endl;
    if (!ALIUtils::IsNumber(valstr)) {
      std::cerr << " ParameterMgr::getVal of an ALIstring that is not a number: " << valstr << std::endl;
      abort();
    }

    //-    std::cout << " getVal " <<  atof( valstr.c_str() ) << " * " << ALIUnitDefinition::GetValueOf(unitstr) << std::endl;
    return atof(valstr.c_str()) * ALIUnitDefinition::GetValueOf(unitstr);
  } else {
    //If there is not a '*', use the dimensionFactor
    if (!ALIUtils::IsNumber(str)) {
      //--- Check if it is referring to a previous parameter.
      ALIdouble val;
      if (getParameterValue(str, val)) {
        return val;
      } else {
        std::cerr << " ParameterMgr::getVal of an string that is not a number nor a previous parameter: " << str
                  << std::endl;
        abort();
      }
    }

    //-    std::cout << "ParameterMgr::getVal " << atof( str.c_str() ) << " * " << dimensionFactor << std::endl;
    return atof(str.c_str()) * dimensionFactor;
  }
}

//----------------------------------------------------------------------------
void ParameterMgr::addParameter(const ALIstring& name, const ALIstring& valstr) {
  if (theParameters.find(name) != theParameters.end()) {
    if (ALIUtils::debug >= 1)
      std::cerr << "!! WARNING: PARAMETER " << name << " appears twice, it will take first value " << std::endl;
  } else {
    theParameters[name] = getVal(valstr);
  }
}

void ParameterMgr::setRandomSeed(const long seed) { CLHEP::HepRandom::setTheSeed(seed); }

//----------------------------------------------------------------------------
void ParameterMgr::addRandomGaussParameter(const ALIstring& name,
                                           const ALIstring& valMean,
                                           const ALIstring& valStdDev) {
  if (theParameters.find(name) != theParameters.end()) {
    if (ALIUtils::debug >= 1)
      std::cerr << "!! WARNING: PARAMETER " << name << " appears twice, it will take first value " << std::endl;
  } else {
    ALIdouble mean = getVal(valMean);
    ALIdouble stddev = getVal(valStdDev);
    ALIdouble val = CLHEP::RandGauss::shoot(mean, stddev);
    theParameters[name] = val;
    if (ALIUtils::debug >= -2)
      std::cout << " addRandomGaussParameter " << name << " " << valMean << " " << valStdDev << " = " << val
                << std::endl;
  }
}

//----------------------------------------------------------------------------
void ParameterMgr::addRandomFlatParameter(const ALIstring& name,
                                          const ALIstring& valMean,
                                          const ALIstring& valInterval) {
  if (theParameters.find(name) != theParameters.end()) {
    if (ALIUtils::debug >= 1)
      std::cerr << "!! WARNING: PARAMETER " << name << " appears twice, it will take first value " << std::endl;
  } else {
    ALIdouble mean = getVal(valMean);
    ALIdouble interval = getVal(valInterval);
    ALIdouble val = CLHEP::HepRandom::getTheEngine()->flat();
    // flat between ]mean-interval, mean+interval[
    val = val * 2 * interval + mean - interval;
    theParameters[name] = val;
    if (ALIUtils::debug >= 2)
      std::cout << " addRandomFlatParameter " << name << " " << valMean << " " << valInterval << " = " << val
                << std::endl;
  }
}

//----------------------------------------------------------------------------
// get the parameter value if parameter name exists and return 1, else return 0
ALIint ParameterMgr::getParameterValue(const ALIstring& name, ALIdouble& val) {
  //-  std::cout << " ParameterMgr::getParameterValu " << name << " " << std::endl;
  //---------- Convert negative parameters
  ALIstring namet = name;
  ALIint negpar = 1;
  if (namet[0] == '-') {
    negpar = -1;
    namet = namet.substr(1, namet.length());
  }

  //---------- Find Parameter by name
  msd::iterator ite = theParameters.find(namet);
  if (ite == theParameters.end()) {
    /*    msd::iterator ite2 = theParameters.find( name );
    for( ite2 = theParameters.begin(); ite2 != theParameters.end(); ite2++ ) {
      std::cout << "PARAMETER: " << (*ite2).first << " = " << (*ite2).second << std::endl;
    }
    */
    return 0;
  } else {
    val = (*ite).second * negpar;
    //-    std::cout << "PARAMETER: " << val << " name " << name << std::endl;
    return 1;
  }
}

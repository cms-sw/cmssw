//-------------------------------------------------
//
//   Class: DTParameter
//
//   Description: Configurable Parameter for Level1 Mu DT Trigger
//
//
//   Author List:
//   C. Grandi
//   Modifications: 
//
//
//--------------------------------------------------

//pz #include "Utilities/Configuration/interface/Architecture.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "L1Trigger/DTUtilities/interface/DTParameter.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//---------------
// C++ Headers --
//---------------

#include <cstdlib>
#include <iostream>

// namespaces
// using namespace edm;
using namespace std;

//----------------
// Constructors --
//----------------
DTParameter::DTParameter() : label_("Undefined"), name_("Undefined") {
  availableValues_.reserve(10);
  currentValue_.clear();
  currentValue_.clear();
}

DTParameter::DTParameter(string label, string name) : 
  label_(label), name_(name) {
  availableValues_.reserve(10);
}

DTParameter::DTParameter(const DTParameter& param) : 
  label_(param.label_), name_(param.name_), currentValue_(param.currentValue_) 
{
  availableValues_.reserve(10);
  ValueConstIterator p;
  for(p=param.availableValues_.begin();p!=param.availableValues_.end();p++) {
    availableValues_.push_back(*p);
  }
}

//--------------
// Destructor --
//--------------
DTParameter::~DTParameter() {
  clear();
}

//--------------
// Operations --
//--------------

void 
DTParameter::addValidParam(string name, double value) {
  DTParameterValue val(name,value);
  availableValues_.push_back(val);
  // Use the first option as default
  if( currentValue_.name()=="Undefined") {
    currentValue_=val;
  }
}

void 
DTParameter::addValidParam(const DTParameterValue& val) { 
  availableValues_.push_back(val);
  // Use the first option as default
  if( currentValue_.name()=="Undefined") {
    currentValue_=val;
  }
}

DTParameter&
DTParameter::operator=(const DTParameter& param) {
  if(this != &param){
    label_=param.label_;
    name_=param.name_;
    currentValue_=param.currentValue_;
    ValueConstIterator p;
    for(p=param.availableValues_.begin();p!=param.availableValues_.end();p++) {
      availableValues_.push_back(*p);
    }
  }
  return *this;
}

DTParameter&
DTParameter::operator=(string val) {

  // Can't assign a numerical value in this way...
  //   note that the following happens when a numerical value is the default
  //   and in .orcarc the parameter is not changed. 
  if(val=="Numerical Value") return *this; 
  
  int allowNumericalValues = 0;
  ValueConstIterator p;
  for(p=availableValues_.begin();p!=availableValues_.end();p++) {
    // check first if there is a match with a non-numerical value
    if(val==(*p).name()) {
      currentValue_=(*p);
      return *this;
    }
    // remember that numerical values are allowed
    if((*p).name()=="Numerical Value") 
      allowNumericalValues = 1;
  }
  // if no matching is found and numerical values are allowed 
  // try to convert the string into a numerical value
  if(allowNumericalValues) {
    currentValue_=DTParameterValue("Numerical Value",
				       atof(val.c_str()));
    return *this;
  }
  
  // no matching is found and numerical values are not allowed
  //cout << "DTParameter: invalid parameter meaning: " << val;
  //cout << " parameter not changed!" << endl;

  // DB: added for gcc-3.2 compatibility
  return *this;
}

void 
DTParameter::clear() {
  label_ = "Undefined";
  name_ = "Undefined";
  availableValues_.clear();
  currentValue_.clear();
}



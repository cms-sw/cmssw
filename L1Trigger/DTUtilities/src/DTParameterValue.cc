//-------------------------------------------------
//
//   Class: DTParameterValue
//
//   Description: Configurable ParameterValue for Level1 Mu DT Trigger
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
#include "L1Trigger/DTUtilities/interface/DTParameterValue.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//---------------
// C++ Headers --
//---------------
#include <cstdlib>


// namespaces
// using namespace edm;
using namespace std;

//----------------
// Constructors --
//----------------
DTParameterValue::DTParameterValue() : 
  name_("Undefined"), value_(-999) {}

DTParameterValue::DTParameterValue(string name, double value) : 
  name_(name), value_(value) {}

DTParameterValue::DTParameterValue(const DTParameterValue& param) :
  name_(param.name_), value_(param.value_) {}

//--------------
// Destructor --
//--------------
DTParameterValue::~DTParameterValue() {}

//--------------
// Operations --
//--------------

DTParameterValue&
DTParameterValue::operator=(const DTParameterValue& param) {
  if(this != &param){
    name_=param.name_;
    value_=param.value_;
  }
  return *this;
}

DTParameterValue&
DTParameterValue::set(string name, double value) {
  name_=name;
  value_=value;
  return *this;
}

void 
DTParameterValue::clear() {
  name_="Undefined";
  value_=0;    
}

//#include "Utilities/Configuration/interface/Architecture.h"

/* \file ExampleClass.cc
 *
 *  $Date: 2005/07/26 10:13:49 $
 *  $Revision: 1.1 $
 *  \author W. Woodpecker - CERN
 */

//#include "Subsystem/Package/interface/ExampleClass.h"
//#include "Subsystem/Package/interface/SomeAlgorithm.h"

using namespace std;

#include "ExampleClass.h"
class SomeAlgorithm {
 public: SomeAlgorithm(){};
};
  


// Constructor
ExampleClass::ExampleClass() :
  theCount(0),
  theAlgo(new SomeAlgorithm()) 
{}

// Destructor
ExampleClass::~ExampleClass(){
  delete theAlgo;
}


// A simple setter
void ExampleClass::setCount(int ticks){
  theCount = ticks;
}


// A simple getter
int ExampleClass::count() const{
  return theCount;
}



// Another setter
void ExampleClass::setValues(const vector<float>& entries) {
  theValues = entries;
}


// A getter returning a const reference
const vector<float>& ExampleClass::values() const {
  return theValues;
}



// A member function
float ExampleClass::computeMean() const {
  float result = 1.;
  //... do all necessary computations...
  return result;
}


//   COCOA class header file
//Id:  OptORisleyPrism.h
//CAT: Model
//
//   Base class to describe Optical Objects of type sensor 2D
// 
//   History: v1.0 
//   Pedro Arce

#ifndef _OPTORISLEYPRISM_HH
#define _OPTORISLEYPRISM_HH

#include "Alignment/CocoaUtilities/interface/CocoaGlobals.h"
#include "Alignment/CocoaModel/interface/OpticalObject.h"
class Measurement;
class LightRay;

class OptORisleyPrism: public OpticalObject
{

public:
  //---------- Constructors / Destructor
  OptORisleyPrism(){ };
  OptORisleyPrism(OpticalObject* parent, const ALIstring& type, const ALIstring& name, const ALIbool copy_data) : 
  OpticalObject( parent, type, name, copy_data){ };
  ~OptORisleyPrism() override{ };

  //---------- Propagate light for measurement meas
  void participateInMeasurement( LightRay& lightray, Measurement& meas, const ALIstring& behav ) override;
  void constructSolidShape() override;


};

#endif


//   COCOA class header file
//Id:  CocoaSolidShape.h
//CAT: Model
//
//   Class to manage the sets of fitted entries (one set per each measurement data set)
// 
//   History: v1.0 
//   Pedro Arce

#ifndef _CocoaSolidShape_HH
#define _CocoaSolidShape_HH

#include "Alignment/CocoaUtilities/interface/CocoaGlobals.h"


class CocoaSolidShape
{

public:
  //---------- Constructors / Destructor
  CocoaSolidShape( ALIstring type );
  virtual ~CocoaSolidShape(){ };

  ALIstring getType() const {
    return theType; }

  //  ALIbool operator==(const CocoaSolidShape& mate );

private:

  ALIstring theType;

};

#endif


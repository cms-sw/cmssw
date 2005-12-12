//   COCOA class header file
//Id:  CocoaSolidShapeBox.h
//
//   History: v1.0 
//   Pedro Arce

#ifndef _CocoaSolidShapeBox_HH
#define _CocoaSolidShapeBox_HH

#include "OpticalAlignment/CocoaDDLObjects/interface/CocoaSolidShape.h"
#include "OpticalAlignment/CocoaUtilities/interface/CocoaGlobals.h"


class CocoaSolidShapeBox : public CocoaSolidShape
{

public:
  //---------- Constructors / Destructor
  CocoaSolidShapeBox( ALIstring type, ALIfloat xdim, ALIfloat ydim, ALIfloat zdim );
  ~CocoaSolidShapeBox(){ };

  ALIfloat getXHalfLength() const {
    return theXHalfLength; }
  ALIfloat getYHalfLength() const {
    return theYHalfLength; }
  ALIfloat getZHalfLength() const {
    return theZHalfLength; }

private:

  ALIfloat theXHalfLength;
  ALIfloat theYHalfLength;
  ALIfloat theZHalfLength;
};

#endif


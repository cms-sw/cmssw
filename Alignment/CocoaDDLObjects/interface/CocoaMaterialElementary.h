//   COCOA class header file
//Id:  CocoaMaterialElementary.h
//CAT: Model
//
//   Class to manage the sets of fitted entries (one set per each measurement data set)
// 
//   History: v1.0 
//   Pedro Arce

#ifndef _CocoaMaterialElementary_HH
#define _CocoaMaterialElementary_HH

#include "Alignment/CocoaUtilities/interface/CocoaGlobals.h"


class CocoaMaterialElementary
{

public:
  //---------- Constructors / Destructor
  CocoaMaterialElementary( ALIstring name, float density, ALIstring symbol, ALIint A, ALIint Z );
  ~CocoaMaterialElementary(){ };

  ALIstring getName() const {
    return theName; }
  float getDensity() const {
    return theDensity; }
  ALIstring getSymbol() const {
    return theSymbol; }
  ALIint getA() const {
    return theA; }
  ALIint getZ() const {
    return theZ; }

  ALIbool operator==(const CocoaMaterialElementary& mate ) const;

private:

  ALIstring theName;
  float theDensity;
  ALIstring theSymbol;
  ALIint theA;
  ALIint theZ;

};

#endif


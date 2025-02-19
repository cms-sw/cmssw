//  COCOA class header file
//Id:  ALIRmDataFromFile.h
//CAT: Model
//
//   Base class for entry data
// 
//   History: Creation 26/06/2005
//   Pedro Arce

#ifndef _ALIRmDataFromFile_HH
#define _ALIRmDataFromFile_HH
#include "Alignment/CocoaUtilities/interface/CocoaGlobals.h"
#include "CLHEP/Vector/Rotation.h"

class ALIRmDataFromFile
{
public:
  //----- Constructor / destructor 
  ALIRmDataFromFile();
  ~ALIRmDataFromFile(){};

 // Access DATA MEMBERS
  ALIbool setAngle( const ALIstring& coord, const ALIdouble val );
  ALIbool setAngleX( const ALIdouble val );
  ALIbool setAngleY( const ALIdouble val );
  ALIbool setAngleZ( const ALIdouble val );
  void constructRm();

  ALIdouble angleX() const { return theAngleX; }
  ALIdouble angleY() const { return theAngleY; }
  ALIdouble angleZ() const { return theAngleZ; }
  CLHEP::HepRotation rm() const { return theRm; }
  ALIstring dataFilled() const { return theDataFilled; }

 // private DATA MEMBERS
private:
  CLHEP::HepRotation theRm;
  ALIdouble theAngleX, theAngleY, theAngleZ;
  ALIstring theDataFilled;
};

#endif

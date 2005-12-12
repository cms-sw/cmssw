#ifndef FittedEntriesReader_h
#define FittedEntriesReader_h
/*---------------------------------------------------------------------------
ClassName:   FittedEntriesReader
Author:      P. Arce
Changes:     24/06/05: creation  
---------------------------------------------------------------------------*/ 
// Description:
// Manages the set of optical objects 


#include "OpticalAlignment/CocoaUtilities/interface/CocoaGlobals.h"
#include "OpticalAlignment/CocoaUtilities/interface/ALIFileIn.h"

class FittedEntriesReader 
{
 private:
  FittedEntriesReader(); 
 public:
  FittedEntriesReader( const ALIstring& filename );
  ~FittedEntriesReader(); 
  ALIbool readFittedEntriesFromFile();
  ALIstring substitutePointBySlash( const ALIstring& nameWithPoints ) const;

 private:
  ALIstring theFileName;
  ALIFileIn  theFile;
  ALIdouble theLengthDim;
  ALIdouble theLengthErrorDim;
  ALIdouble theAngleDim;
  ALIdouble theAngleErrorDim;
};

#endif

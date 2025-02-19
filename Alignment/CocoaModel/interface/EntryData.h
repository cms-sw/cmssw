//  COCOA class header file
//Id:  EntryData.h
//CAT: Model
//
//   Base class for entry data
// 
//   History: v1.0 
//   Pedro Arce

#ifndef _EntryData_HH
#define _EntryData_HH
#include "Alignment/CocoaUtilities/interface/CocoaGlobals.h"
#include <vector>

class EntryData
{
  friend std::ostream& operator << (std::ostream& os, const EntryData& c);
  
public:
  //----- Constructor / destructor 
  EntryData();
  ~EntryData();

  void fill( const std::vector<ALIstring>& wordlist );
 
 // Access DATA MEMBERS
  const ALIstring& longOptOName() const { return fLongOptOName; }
  const ALIstring& shortOptOName() const { return fShortOptOName; }
  const ALIstring& optOName() const { return longOptOName(); }
  const ALIstring& entryName() const { return fEntryName; }  
  ALIdouble valueOriginal() const { return fValueOriginal; }
  ALIdouble valueDisplacement() const { return fValueDisplacement; }
  ALIdouble sigma() const { return fSigma; }
  ALIint quality() const { return fQuality; }
  void setValueDisplacement( const ALIdouble val ) 
    { fValueDisplacement = val; }
  //-  ALIint fitPos() const { return fFitPos; }

private:
 // private DATA MEMBERS
protected:
  ALIstring fLongOptOName;
  ALIstring fShortOptOName;
  ALIstring fEntryName;
  ALIdouble fValueOriginal;
  ALIdouble fValueDisplacement;
  ALIdouble fSigma;
  ALIuint fQuality;
  //- ALIint fFitPos;

};

#endif

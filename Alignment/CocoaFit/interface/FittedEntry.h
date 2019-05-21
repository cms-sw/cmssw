//   COCOA class header file
//Id:  FittedEntry.h
//CAT: Model
//
//   Class to store the data of a fitted entry (only those of quality 'unk')
//
//   History: v1.0
//   Pedro Arce

#ifndef FittedEntry_HH
#define FittedEntry_HH

#include "Alignment/CocoaUtilities/interface/CocoaGlobals.h"
#include <vector>
class Entry;

class FittedEntry {
public:
  //---------- Constructors / Destructor
  FittedEntry(){};
  FittedEntry(Entry* entry, ALIint order, ALIdouble sigma);
  FittedEntry(ALIstring name, float value, float sigma);
  FittedEntry(const std::vector<FittedEntry*>& vFEntry);
  ~FittedEntry(){};

  void BuildName();

  ALIstring getOptOName() const { return theOptOName; }
  ALIstring getEntryName() const { return theEntryName; }
  ALIstring getName() const { return theName; }
  ALIdouble getValue() const { return theValue; }
  ALIdouble getSigma() const { return theSigma; }
  ALIdouble getOrigValue() const { return theOrigValue; }
  ALIdouble getOrigSigma() const { return theOrigSigma; }
  ALIint getOrder() const { return theOrder; }
  ALIint getQuality() const { return theQuality; }
  Entry* getEntry() const { return theEntry; }

private:
  ALIdouble theValue;
  Entry* theEntry;
  ALIint theOrder;
  ALIstring theName;
  ALIstring theOptOName;
  ALIstring theEntryName;
  ALIdouble theSigma;
  ALIdouble theOrigValue;
  ALIdouble theOrigSigma;
  ALIint theQuality;
};

#endif

//   COCOA class header file
//Id:  HistoDef.h
//CAT: Model
//
//   Class to store the data of a fitted entry (only those of quality 'unk')
//
//   History: v1.0
//   Pedro Arce

#ifndef HistoDef_HH
#define HistoDef_HH

#include "Alignment/CocoaUtilities/interface/CocoaGlobals.h"
#include <vector>
class Entry;

class HistoDef {
public:
  //---------- Constructors / Destructor
  HistoDef(){};
  void init(ALIstring name);
  ~HistoDef(){};

  ALIstring name() const { return theName; }
  float minimum() const { return theMin; }
  float maximum() const { return theMax; }

  void setMinimum(float min) { theMin = min; }
  void setMaximum(float max) { theMax = max; }

private:
  ALIstring theName;
  float theMin;
  float theMax;
};

#endif

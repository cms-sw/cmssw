//   COCOA class header file
//Id:  EntryNoDim.h
//CAT: Model
//
//   class for entries that have dimension of length
// 
//   History: v1.0 
//   Pedro Arce

#ifndef _ENTRYNoDim_HH
#define _ENTRYNoDim_HH

#include "Alignment/CocoaModel/interface/Entry.h"

class EntryNoDim : public Entry
{
public:
  //-  EntryNoDim(){ };
  EntryNoDim( const ALIstring type ): Entry(type){ 
    theDimType = ED_nodim;
    //std::cout << "entryNoDim" << std::endl;
};
  ~EntryNoDim(){};

 // Access DATA MEMBERS
  //----------- Return value and sigma dimension factors (1. as object of this class have no dimension)
  virtual ALIdouble ValueDimensionFactor() const{
    return _ValueDimensionFactor;
  }
  virtual ALIdouble SigmaDimensionFactor() const{
    return _SigmaDimensionFactor;
  }
  //----- Return starting displacement for derivative
  virtual ALIdouble startingDisplacement() {
     return _startingDisplacement;
  }


private:
 // static DATA MEMBERS
  static ALIdouble _ValueDimensionFactor;
  static ALIdouble _SigmaDimensionFactor;
  static ALIdouble _startingDisplacement;
};

#endif








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

class EntryNoDim : public Entry {
public:
  //-  EntryNoDim(){ };
  EntryNoDim(const ALIstring type) : Entry(type) {
    theDimType = ED_nodim;
    //std::cout << "entryNoDim" << std::endl;
  };
  ~EntryNoDim() override{};

  // Access DATA MEMBERS
  //----------- Return value and sigma dimension factors (1. as object of this class have no dimension)
  ALIdouble ValueDimensionFactor() const override { return 1.0; }
  ALIdouble SigmaDimensionFactor() const override { return 1.0; }
  //----- Return starting displacement for derivative
  ALIdouble startingDisplacement() override { return 0.1; }
};

#endif

//   COCOA class header file
//Id:  EntryLength.h
//CAT: Model
//
//   class for entries that have dimension of length
//
//   History: v1.0
//   Pedro Arce

#ifndef _ENTRYLENGTH_HH
#define _ENTRYLENGTH_HH

#include "Alignment/CocoaModel/interface/Entry.h"
#include "Alignment/CocoaUtilities/interface/ALIUtils.h"

class EntryLength : public Entry {
public:
  //-  EntryLength(){ };
  EntryLength(const ALIstring& type) : Entry(type) {
    // std::cout << "entrylength" << std::endl;
    theDimType = ED_length;
  };
  ~EntryLength() override{};

  //----- Return value and sigma dimension factors
  ALIdouble ValueDimensionFactor() const override { return ALIUtils::LengthValueDimensionFactor(); }
  ALIdouble SigmaDimensionFactor() const override { return ALIUtils::LengthSigmaDimensionFactor(); }
  ALIdouble OutputValueDimensionFactor() const override { return ALIUtils::OutputLengthValueDimensionFactor(); }
  ALIdouble OutputSigmaDimensionFactor() const override { return ALIUtils::OutputLengthSigmaDimensionFactor(); }

  //----- Return starting displacement for derivative
  ALIdouble startingDisplacement() override { return _startingDisplacement; }

private:
  // static DATA MEMBERS
  static ALIdouble _startingDisplacement;
};

#endif

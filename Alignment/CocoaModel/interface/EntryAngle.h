//   COCOA class header file
//Id:  EntryAngle.h
//CAT: Model
//
//   class for entries that have dimension of length
//
//   History: v1.0
//   Pedro Arce

#ifndef _ENTRYANGLE_HH
#define _ENTRYANGLE_HH

#include "Alignment/CocoaModel/interface/Entry.h"
#include "Alignment/CocoaUtilities/interface/ALIUtils.h"

class EntryAngle : public Entry {
public:
  //-  EntryAngle(){ };
  EntryAngle(const ALIstring& type) : Entry(type) {
    theDimType = ED_angle;
    //-    std::cout << "entryangle " << type << std::endl;
  };
  ~EntryAngle() override{};

  //----- Return value and sigma dimension factors
  ALIdouble ValueDimensionFactor() const override { return ALIUtils::AngleValueDimensionFactor(); }
  ALIdouble SigmaDimensionFactor() const override { return ALIUtils::AngleSigmaDimensionFactor(); }
  ALIdouble OutputValueDimensionFactor() const override { return ALIUtils::OutputAngleValueDimensionFactor(); }
  ALIdouble OutputSigmaDimensionFactor() const override { return ALIUtils::OutputAngleSigmaDimensionFactor(); }

  //----- Return starting displacement for derivative
  ALIdouble startingDisplacement() override { return _startingDisplacement; }

private:
  // static DATA MEMBERS
  //----------- Factor by which you multiply a value to get it in radians
  static ALIdouble _startingDisplacement;
};

#endif

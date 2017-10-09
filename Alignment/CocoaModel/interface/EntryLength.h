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


class EntryLength : public Entry
{
public:
  //-  EntryLength(){ };
  EntryLength( const ALIstring& type ): Entry(type){
    // std::cout << "entrylength" << std::endl;
    theDimType = ED_length;
  };
  ~EntryLength(){};

 //----- Return value and sigma dimension factors
  virtual ALIdouble ValueDimensionFactor() const{
    return ALIUtils::LengthValueDimensionFactor(); 
  }
  virtual ALIdouble SigmaDimensionFactor() const{
    return ALIUtils::LengthSigmaDimensionFactor(); 
  }
  virtual ALIdouble OutputValueDimensionFactor() const{
    return ALIUtils::OutputLengthValueDimensionFactor(); 
  }
  virtual ALIdouble OutputSigmaDimensionFactor() const{
    return ALIUtils::OutputLengthSigmaDimensionFactor(); 
  }

  //----- Return starting displacement for derivative
  virtual ALIdouble startingDisplacement() {
     return _startingDisplacement;
  }

 private: 
 // static DATA MEMBERS
  static ALIdouble _startingDisplacement;
};

#endif


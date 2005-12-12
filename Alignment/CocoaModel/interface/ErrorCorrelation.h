//   COCOA class header file
//Id:  ErrorCorrelation.h
//CAT: Model
//
//   Error correlation: contains first and second entry and correlation value
// 
//   History: v1.0  29/01/02   Pedro Arce

#ifndef ErrorCorrelation_h
#define ErrorCorrelation_h

#include "OpticalAlignment/CocoaUtilities/interface/CocoaGlobals.h" 

typedef std::pair<ALIstring, ALIstring> pss;

class ErrorCorrelation {

 public:
  ErrorCorrelation( const pss& entry1, const pss& entry2, const ALIdouble corr ): theEntry1(entry1), theEntry2(entry2), theCorr(corr) {}

  const pss& entry1() const {
    return theEntry1; }
  const pss& entry2() const {
    return theEntry2; }
  const ALIdouble correlation() const {
    return theCorr; }

 private:
  pss theEntry1;
  pss theEntry2;
  ALIdouble theCorr;

};

#endif

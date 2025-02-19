//   COCOA class header file
//Id:  ErrorCorrelation.h
//CAT: Model
//
//   Error correlation: contains first and second entry and correlation value
// 
//   History: v1.0  29/01/02   Pedro Arce

#ifndef ErrorCorrelation_h
#define ErrorCorrelation_h

#include "Alignment/CocoaUtilities/interface/CocoaGlobals.h" 

typedef std::pair<ALIstring, ALIstring> pss;

class ErrorCorrelation {

 public:
  ErrorCorrelation( const pss& entry1, const pss& entry2, const ALIdouble corr );

  void update( const ALIdouble corr );

  const pss& getEntry1() const {
    return theEntry1; }
  const pss& getEntry2() const {
    return theEntry2; }
  const ALIdouble getCorrelation() const {
    return theCorr; }

 private:
  pss theEntry1;
  pss theEntry2;
  ALIdouble theCorr;

};

#endif

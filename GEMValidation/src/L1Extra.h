#ifndef GEMValidation_L1Extra_h
#define GEMValidation_L1Extra_h

#include "GEMCode/GEMValidation/src/GMTCand.h"

class L1Extra : public GMTCand
{
 public:
  /// constructor
  L1Extra();
  /// copy constructor
  L1Extra(const L1Extra&);
  /// destructor
  ~L1Extra();

  /*  
  init(const L1MuRegionalCand *t, CSCTFPtLUT* ptLUT, 
       edm::ESHandle< L1MuTriggerScales > &muScales, 
       edm::ESHandle< L1MuTriggerPtScale > &muPtScale);
  */
  //  const L1MuionalCand * l1cand;
  GMTCand* gmtCand();
  
 private:
  double pt_;
  double eta_;
  double phi_;
  double dr_;
};

#endif

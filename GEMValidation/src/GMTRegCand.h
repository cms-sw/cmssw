#ifndef GEMValidation_GMTRegCand_h
#define GEMValidation_GMTRegCand_h

#include "GEMCode/GEMValidation/src/TFCand.h"

class GMTRegCand
{
 public:
  /// constructor
  GMTRegCand();
  /// copy constructor
  GMTRegCand(const GMTRegCand&);
  /// destructor
  ~GMTRegCand();

  /*  
  init(const L1MuRegionalCand *t, CSCTFPtLUT* ptLUT, 
       edm::ESHandle< L1MuTriggerScales > &muScales, 
       edm::ESHandle< L1MuTriggerPtScale > &muPtScale);
  */
  //  const L1MuRegionalCand * l1reg;
  TFCand* tfCand();
  double pt() const {return pt_;}
  double eta() const {return eta_;}
  double phi() const {return phi_;}
  double dr() const {return dr_;}
  void print();
  
 private:
  double pt_;
  double eta_;
  double phi_;
  double dr_;
};

#endif

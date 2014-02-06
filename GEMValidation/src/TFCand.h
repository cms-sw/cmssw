#ifndef GEMValidation_TFCand_h
#define GEMValidation_TFCand_h

#include "GEMCode/GEMValidation/src/TFTrack.h"

class TFCand
{
 public:
  /// constructor
  TFCand();
  /// copy constructor
  TFCand(const TFCand&);
  /// destructor
  ~TFCand();

  void init(const L1MuRegionalCand *t, CSCTFPtLUT* ptLUT, 
	    edm::ESHandle< L1MuTriggerScales > &muScales, 
 	    edm::ESHandle< L1MuTriggerPtScale > &muPtScale); 

  const L1MuRegionalCand * l1Cand() const {return l1Cand_;}

  TFTrack* tftrack() const {return tftrack_;}

  std::vector < CSCDetId > ids() const {return ids_;}

  double pt() const {return pt_;}
  double eta() const {return eta_;}
  double phi() const {return phi_;}
  double dr() const {return dr_;}
  
 private:

  const L1MuRegionalCand * l1Cand_;
  TFTrack* tftrack_;
  std::vector<CSCDetId> ids_;
  double pt_;
  double eta_;
  double phi_;
  double dr_;
  unsigned nTFStubs;
};

#endif

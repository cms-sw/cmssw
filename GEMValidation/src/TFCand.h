#ifndef GEMValidation_TFCand_h
#define GEMValidation_TFCand_h

class TFCand : TFTrack
{
 public:
  TFCand();
  ~TFCand();

  /*  
  init(const L1MuRegionalCand *t, CSCTFPtLUT* ptLUT, 
       edm::ESHandle< L1MuTriggerScales > &muScales, 
       edm::ESHandle< L1MuTriggerPtScale > &muPtScale);
  */
  //  const L1MuRegionalCand * l1cand;
  TFTrack* tftrack();
  std::vector < CSCDetId > ids();
  
 private:
  double pt_;
  double eta_;
  double phi_;
  double dr_;
  
};

#endif

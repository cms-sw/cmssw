#ifndef UEAnalysisCorrCali_h
#define UEAnalysisCorrCali_h

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

class UEAnalysisCorrCali{
 public :
  
  UEAnalysisCorrCali();
  ~UEAnalysisCorrCali(){}

  float calibrationPt(float ptReco,std::string tkpt);
  float correctionPtTrans(float ptReco,std::string tkpt);
  float correctionPtToward(float ptReco,std::string tkpt);
  float correctionPtAway(float ptReco,std::string tkpt);
  
  float correctionNTrans(float ptReco,std::string tkpt);
  float correctionNToward(float ptReco,std::string tkpt);
  float correctionNAway(float ptReco,std::string tkpt);
};

#endif

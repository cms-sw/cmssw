#ifndef UEAnalysisCorrCali_h
#define UEAnalysisCorrCali_h

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

class UEAnalysisCorrCali{
 public :
  
  UEAnalysisCorrCali();
  ~UEAnalysisCorrCali(){}

  float calibrationPt(float ptReco,string tkpt);
  float correctionPtTrans(float ptReco,string tkpt);
  float correctionPtToward(float ptReco,string tkpt);
  float correctionPtAway(float ptReco,string tkpt);
  
  float correctionNTrans(float ptReco,string tkpt);
  float correctionNToward(float ptReco,string tkpt);
  float correctionNAway(float ptReco,string tkpt);
};

#endif

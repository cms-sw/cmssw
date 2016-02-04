#ifndef TPNCor_H
#define TPNCor_H

#include "TObject.h"
#include <map>

class TPNCor: public TObject 
{
  
 private:
  
  
 

 public:
  // Default Constructor, mainly for Root
  TPNCor(std::string filename);
  
  // Destructor: Does nothing
  virtual ~TPNCor();
  
  enum VarGain  { iGain0, iGain1, iSizeGain }; 
  enum VarParPN  { iPar0, iPar1, iPar2, iSizePar }; 

  double getPNCorrectionFactor( double val0 , int gain );

  // Declaration for PN linearity corrections
  double corParams[iSizeGain][iSizePar];
  int isFileOK;
  
  //  ClassDef(TPNCor,1)
    


    };
    
#endif

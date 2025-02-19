#include "UEAnalysisCorrCali.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TVector3.h>

#include <vector>
#include <math.h>

UEAnalysisCorrCali::UEAnalysisCorrCali()
{
  std::cout << "UEAnalysisCorrCali constructor " <<std::endl;
}

float UEAnalysisCorrCali::calibrationPt(float ptReco,std::string tkpt){
  if(tkpt=="900"){
    float corr = 0.1122*exp(-(0.2251*ptReco))+1.086-0.0005408*ptReco;
    return  corr;
  }
  if(tkpt=="500"){
    float corr = 0.1389*exp(-(0.2364*ptReco))+1.048-0.0001663*ptReco;
    return  corr;
  }
}

float UEAnalysisCorrCali::correctionPtTrans(float ptReco,std::string tkpt){
  if(tkpt=="900"){
    float corr = 1.214*exp(-(0.9637*ptReco))+1.204-0.0003461*ptReco;
    return  corr;
  }
  if(tkpt=="500"){
    float corr = 0.4174*exp(-(0.537*ptReco))+1.136-0.0001166*ptReco;
    return  corr;
  }
}

float UEAnalysisCorrCali::correctionPtToward(float ptReco,std::string tkpt){
  if(tkpt=="900"){
    float corr = 0.1037*exp(-(0.1382*ptReco))+1.117-0.0006322*ptReco;
    return  corr;
  }
  if(tkpt=="500"){
    float corr = 0.166*exp(-(0.1989*ptReco))+1.073-0.000245*ptReco;
    return  corr;
  }
}

float UEAnalysisCorrCali::correctionPtAway(float ptReco,std::string tkpt){
  if(tkpt=="900"){
    float corr = 0.2707*exp(-(0.2685*ptReco))+1.169-0.000411*ptReco;
    return  corr;
  }
  if(tkpt=="500"){
    float corr = 0.2835*exp(-(0.2665*ptReco))+1.1-0.0001659*ptReco;
    return  corr;
  }
}

float UEAnalysisCorrCali::correctionNTrans(float ptReco,std::string tkpt){
  if(tkpt=="900"){
    float corr = 1.101*exp(-(0.9939*ptReco))+1.198-0.0001467*ptReco;
    return  corr;
  }
  if(tkpt=="500"){
    float corr = 0.3322*exp(-(0.445*ptReco))+1.146+0.00002659*ptReco;
    return  corr;
  }
}

float UEAnalysisCorrCali::correctionNToward(float ptReco,std::string tkpt){
  if(tkpt=="900"){
    float corr = 0.9264*exp(-(1.053*ptReco))+1.16-0.0005176*ptReco;
    return  corr;
  }
  if(tkpt=="500"){
    float corr = 0.2066*exp(-(0.3254*ptReco))+1.109-0.00006666*ptReco;
    return  corr;
  }
}

float UEAnalysisCorrCali::correctionNAway(float ptReco,std::string tkpt){
  if(tkpt=="900"){
    float corr = 0.2663*exp(-(0.342*ptReco))+1.178-0.0004006*ptReco;
    return  corr;
  }
  if(tkpt=="500"){
    float corr = 0.316*exp(-(0.3741*ptReco))+1.136-0.0002407*ptReco;
    return  corr;
  }
}

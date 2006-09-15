#include "CalibCalorimetry/HcalTPGAlgos/interface/HcaluLUTTPGCoder.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <iostream>
#include <fstream>

static const int LUT_SIZE = 128;

HcaluLUTTPGCoder::HcaluLUTTPGCoder(const char* filename) {
  loadLUTs(filename);
}


void HcaluLUTTPGCoder::loadLUTs(const char* filename){
  int tool;
  std::ifstream userfile;
  userfile.open(filename);

  if( userfile ) {
    int nluts;
    std::vector<int> loieta,hiieta;
    userfile >> nluts;

    luts_.resize(nluts);
    for (int i=0; i<nluts; i++) {
      luts_[i].resize(LUT_SIZE); 
    }
    
    for (int i=0; i<nluts; i++) {
      userfile >> tool;
      loieta.push_back(tool);
    }
    for (int i=0; i<nluts; i++) {
      userfile >> tool;
      hiieta.push_back(tool);
    }
    
    for (int j=0; j<LUT_SIZE; j++) { 
      for(int i = 0; i <nluts; i++) {
	  userfile >> luts_[i][j];}
    }
    
    userfile.close();
    /*
    std::cout << nluts << std::endl;

    for(int i = 0; i <nluts; i++) {
      for (int j=0; j<LUT_SIZE; j++) { 
	std::cout << i << "[" << j << "] = "<< luts_[i][j] << std::endl;
      }
    }
    */

    // map |ieta| to LUT
    for (int j=1; j<=41; j++) {
      int ilut=-1;
      for (ilut=0; ilut<nluts; ilut++)
	if (j>=loieta[ilut] && j<=hiieta[ilut]) break;
      if (ilut==nluts) {
	ietaLutMap_[j-1]=0;
	// TODO: log warning
      }
      else ietaLutMap_[j-1]=&(luts_[ilut]);
      //      std::cout << j << "->" << ilut << std::endl;
    }    
  }

}

void HcaluLUTTPGCoder::adc2Linear(const HBHEDataFrame& df, IntegerCaloSamples& ics) const{
  const InputLUT* lut=ietaLutMap_[df.id().ietaAbs()-1];
  if (lut==0) {
    // log warning!
  } else {
    for (int i=0; i<df.size(); i++)
      ics[i]=(*lut)[df[i].adc()];
  }
}

void HcaluLUTTPGCoder::adc2Linear(const HFDataFrame& df, IntegerCaloSamples& ics)  const{
  const InputLUT* lut=ietaLutMap_[df.id().ietaAbs()-1];
  if (lut==0) {
    // log warning!
  } else {
    for (int i=0; i<df.size(); i++)
      ics[i]=(*lut)[df[i].adc()];
  }
}


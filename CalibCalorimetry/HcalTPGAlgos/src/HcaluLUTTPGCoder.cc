#include "CalibCalorimetry/HcalTPGAlgos/interface/HcaluLUTTPGCoder.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <iostream>
#include <fstream>

static const int INPUT_LUT_SIZE = 128;

HcaluLUTTPGCoder::HcaluLUTTPGCoder(const char* filename) {
  loadILUTs(filename);
}

HcaluLUTTPGCoder::HcaluLUTTPGCoder(const char* filename, const char* fname2) {
  loadILUTs(filename);
  loadOLUTs(fname2);
}

void HcaluLUTTPGCoder::loadILUTs(const char* filename){
  int tool;
  std::ifstream userfile;
  userfile.open(filename);

  if( userfile ) {
    int nluts;
    std::vector<int> loieta,hiieta;
    userfile >> nluts;

    inputluts_.resize(nluts);
    for (int i=0; i<nluts; i++) {
      inputluts_[i].resize(INPUT_LUT_SIZE); 
    }
    
    for (int i=0; i<nluts; i++) {
      userfile >> tool;
      loieta.push_back(tool);
    }
    for (int i=0; i<nluts; i++) {
      userfile >> tool;
      hiieta.push_back(tool);
    }
    
    for (int j=0; j<INPUT_LUT_SIZE; j++) { 
      for(int i = 0; i <nluts; i++) {
	  userfile >> inputluts_[i][j];}
    }
    
    userfile.close();
    /*
    std::cout << nluts << std::endl;

    for(int i = 0; i <nluts; i++) {
      for (int j=0; j<INPUT_LUT_SIZE; j++) { 
	std::cout << i << "[" << j << "] = "<< inputluts_[i][j] << std::endl;
      }
    }
    */

    // map |ieta| to LUT
    for (int j=1; j<=41; j++) {
      int ilut=-1;
      for (ilut=0; ilut<nluts; ilut++)
	if (j>=loieta[ilut] && j<=hiieta[ilut]) break;
      if (ilut==nluts) {
	ietaILutMap_[j-1]=0;
	// TODO: log warning
      }
      else ietaILutMap_[j-1]=&(inputluts_[ilut]);
      //      std::cout << j << "->" << ilut << std::endl;
    }    
  }
}

void HcaluLUTTPGCoder::loadOLUTs(const char* filename) {
  static const int LUT_SIZE=1024;
  int tool;
  std::ifstream userfile;
  userfile.open(filename);

  if( userfile ) {
    int nluts;
    std::vector<int> loieta,hiieta;
    userfile >> nluts;

    outputluts_.resize(nluts);
    for (int i=0; i<nluts; i++) {
      outputluts_[i].resize(LUT_SIZE); 
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
	  userfile >> outputluts_[i][j];}
    }
    
    userfile.close();
    /*    
    std::cout << nluts << std::endl;

    for(int i = 0; i <nluts; i++) {
      for (int j=0; j<LUT_SIZE; j++) { 
	std::cout << i << "[" << j << "] = "<< outputluts_[i][j] << std::endl;
      }
    }
    */

    // map |ieta| to LUT
    static const int N_TOWER=32;
    for (int j=1; j<=N_TOWER; j++) {
      int ilut=-1;
      for (ilut=0; ilut<nluts; ilut++)
	if (j>=loieta[ilut] && j<=hiieta[ilut]) break;
      if (ilut==nluts) {
	ietaOLutMap_[j-1]=0;
	// TODO: log warning
      }
      else ietaOLutMap_[j-1]=&(outputluts_[ilut]);
      //      std::cout << j << "->" << ilut << std::endl;
    }    
  } else {
    throw cms::Exception("Invalid Data") << "Unable to read " << filename;
  }
}


void HcaluLUTTPGCoder::adc2Linear(const HBHEDataFrame& df, IntegerCaloSamples& ics) const{
  const LUTType* lut=ietaILutMap_[df.id().ietaAbs()-1];
  if (lut==0) {
    throw cms::Exception("Missing Data") << "No LUT for " << df.id();
  } else {
    for (int i=0; i<df.size(); i++) {
      if (df[i].adc() >= INPUT_LUT_SIZE)
	throw cms::Exception("ADC overflow for tower:") << i << " adc= " << df[i].adc();
      ics[i]=(*lut)[df[i].adc()];
      //      std::cout << df.id() << '[' << i <<']' << df[i].adc() << "->" << ics[i] << std::endl;
    }
  }
}

void HcaluLUTTPGCoder::adc2Linear(const HFDataFrame& df, IntegerCaloSamples& ics)  const{
  const LUTType* lut=ietaILutMap_[df.id().ietaAbs()-1];
  if (lut==0) {
    throw cms::Exception("Missing Data") << "No LUT for " << df.id();
  } else {
    for (int i=0; i<df.size(); i++){
      if (df[i].adc() >= INPUT_LUT_SIZE)
      throw cms::Exception("ADC overflow for HF tower:") << i << " adc= " << df[i].adc();
      ics[i]=(*lut)[df[i].adc()];
    }
  }
}

void HcaluLUTTPGCoder::compress(const IntegerCaloSamples& ics, const std::vector<bool>& featureBit, HcalTriggerPrimitiveDigi& tp) const {
  HcalTrigTowerDetId id(ics.id());
  tp=HcalTriggerPrimitiveDigi(id);
  tp.setSize(ics.size());
  tp.setPresamples(ics.presamples());

  int itower=id.ietaAbs();
  const LUTType* lut=ietaOLutMap_[itower-1];
  if (lut==0) {
    throw cms::Exception("Invalid Data") << "No LUT available for " << itower;
  } 

  for (int i=0; i<ics.size(); i++) {
    int sample=ics[i];
    if (sample>=int(lut->size())) {
      // throw cms::Exception("Out of Range") << "LUT has " << lut->size() << " entries for " << itower << " but " << sample << " was requested.";
      sample=lut->size()-1;
    }
    tp.setSample(i,HcalTriggerPrimitiveSample((*lut)[sample],featureBit[i],0,0));
    //  std::cout << id << ":" << sample << "-->" << (*lut)[sample] << std::endl;
  }    

}

#include "CalibCalorimetry/HcalTPGAlgos/interface/HcaluLUTTPGCoder.h"
#include "Geometry/HcalTowerAlgo/src/HcalHardcodeGeometryData.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include <iostream>
#include <fstream>
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
// Geometry
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalHardcodeGeometryLoader.h"
#include "Geometry/HcalTowerAlgo/interface/CaloTowerHardcodeGeometryLoader.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"


static const int INPUT_LUT_SIZE = 128;
static const  int nluts= 6912;

HcaluLUTTPGCoder::HcaluLUTTPGCoder(const char* filename) {
  generateILUTs();
  loadILUTs(filename);
}

HcaluLUTTPGCoder::HcaluLUTTPGCoder(const char* filename, const char* fname2) {
  loadILUTs(filename);
  loadOLUTs(fname2);
}
void HcaluLUTTPGCoder::LUTmemory() {

      
    inputluts_.resize(nluts);
    for (int i=0; i<nluts; i++) {
    inputluts_[i].resize(INPUT_LUT_SIZE); 
    }
    

}
/*
void HcaluLUTTPGCoder::LUTwrite(const int i, const int j, const int k){

  //  inputluts_[i][j]=k;

}
*/

void HcaluLUTTPGCoder::getConditions(const edm::EventSetup& es) const {
//
// Using Jeremy's template
//
  edm::ESHandle<HcalDbService> conditions;
  es.get<HcalDbRecord>().get(conditions);
  const HcalQIEShape* shape = conditions->getHcalShape ();

  HcalCalibrations calibrations;
  HcalTopology theCaloTopo;
 
  
    int detId[nluts];

  float adc2fC_[128];
  //short unsigned int lin_lut_val[128];
  float rechit_calib = 1;                                                                          
                                           
  //std::vector<std::pair<int,short unsigned int> >::iterator iter2;

														       
    HcalHardcodeGeometryLoader loader(theCaloTopo);
    std::auto_ptr<CaloSubdetectorGeometry> hcalGeometry = loader.load();
 
    CaloGeometry geometry;
    geometry.setSubdetGeometry(DetId::Hcal, HcalBarrel, hcalGeometry.get());
    geometry.setSubdetGeometry(DetId::Hcal, HcalEndcap, hcalGeometry.get());
    geometry.setSubdetGeometry(DetId::Hcal, HcalForward, hcalGeometry.get());
 
    CaloTowerHardcodeGeometryLoader towerLoader;
    std::auto_ptr<CaloSubdetectorGeometry> towerGeometry = towerLoader.load();
    geometry.setSubdetGeometry(DetId::Calo, 1, towerGeometry.get()); 
	  										       
    std::vector<DetId>::const_iterator detItr;
    std::vector<DetId> hbDets = geometry.getValidDetIds(DetId::Hcal, HcalBarrel);
    std::vector<DetId> heDets = geometry.getValidDetIds(DetId::Hcal, HcalEndcap);
    std::vector<DetId> hfDets = geometry.getValidDetIds(DetId::Hcal, HcalForward);

    // std::vector<std::pair<int,short unsigned int> >myvec;
    int count=0;
    for(detItr = hbDets.begin(); detItr != hbDets.end(); ++detItr) {

	int phi_ = HcalDetId(*detItr).iphi();
	int eta_ = HcalDetId(*detItr).ieta();
	HcalDetId cell(*detItr);
	conditions->makeHcalCalibration (cell, &calibrations);
	const HcalQIECoder* channelCoder = conditions->getHcalCoder (cell);
	HcalCoderDb coder (*channelCoder, *shape);
	int rawdetId_ = detItr->rawId();
        detId[count]=rawdetId_;

	//std::cout << "id: " << detItr->rawId() << " eta: " << eta_ << " phi:" << phi_ << std::endl;
     
	float ped_ = (calibrations.pedestal(0)+calibrations.pedestal(1)+calibrations.pedestal(2)+calibrations.pedestal(3))/4;
	float gain_= (calibrations.gain(0)+calibrations.gain(1)+calibrations.gain(2)+calibrations.gain(3))/4;                  
                                                                                       
	// do the HB digis
	HBHEDataFrame frame(cell);
	frame.setSize(1);
           
	CaloSamples samples(cell, 1);
	for (int j = 0; j <= 0x7F; j++) {
	  HcalQIESample adc(j);
	  frame.setSample(0,adc);
	  coder.adc2fC(frame,samples);
        
          adc2fC_[j] = samples[0];
         
          
	  //LUTwrite(count,j,int(std::min(std::max(0,int((adc2fC_[j] - ped_)/1.)), 0x7F))); 
	          
	}
	count++;
        
      }


    for(detItr = heDets.begin(); detItr != heDets.end(); ++detItr) {
     
      int phi_ = HcalDetId(*detItr).iphi();
      int eta_ = HcalDetId(*detItr).ieta();
      HcalDetId cell(*detItr);
      conditions->makeHcalCalibration (cell, &calibrations);
      const HcalQIECoder* channelCoder = conditions->getHcalCoder (cell);
      HcalCoderDb coder (*channelCoder, *shape);
      int rawdetId_ = detItr->rawId();
      detId[count]=rawdetId_;
      float ped_ = (calibrations.pedestal(0)+calibrations.pedestal(1)+calibrations.pedestal(2)+calibrations.pedestal(3))/4;
      float gain_= (calibrations.gain(0)+calibrations.gain(1)+calibrations.gain(2)+calibrations.gain(3))/4;
      // do the HE digis
      HBHEDataFrame frame(cell);
      frame.setSize(1);

      CaloSamples samples(cell, 1);
      for (int j = 0; j <= 0x7F; j++) {
        HcalQIESample adc(j);
        frame.setSample(0,adc);
        coder.adc2fC(frame,samples);
                                                                                                                     
	adc2fC_[j] = samples[0];
	/*
	if (abs(eta_) < 21) 
         
	 LUTwrite(count,j,std::min(std::max(0,int((adc2fC_[j] - ped_)/1.)), 0x7F));
	
	else if  (abs(eta_) < 27)  LUTwrite(count,j,std::min(std::max(0,int((adc2fC_[j] - ped_)/2.)), 0x7F));
            
        else {
	   LUTwrite(count,j,std::min(std::max(0,int((adc2fC_[j] - ped_)/5.)), 0x7F));	  
	  }
	*/

	//	myvec.push_back(std::make_pair(rawdetId_,lin_lut_val[j]));
      }
      count++;
      
    }

       float cosheta_[41], lsb_ = 1./16.;
       for (int i = 0; i < 13; i++) {
	 std::cout << "eta bound: " << theHFEtaBounds[i] << " " << theHFEtaBounds[i+1] << std::endl; 
	 cosheta_[i+29] = cosh((theHFEtaBounds[i+1] + theHFEtaBounds[i])/2.);
       }
      
       // now do HF
       for(detItr = hfDets.begin(); detItr != hfDets.end(); ++detItr) {
	
	 int phi_ = HcalDetId(*detItr).iphi();
	 int eta_ = HcalDetId(*detItr).ieta();
	 HcalDetId cell(*detItr);
	 conditions->makeHcalCalibration (cell, &calibrations);
	 const HcalQIECoder* channelCoder = conditions->getHcalCoder (cell);
	 HcalCoderDb coder (*channelCoder, *shape);                                                                                  
	 int rawdetId_ = detItr->rawId(); 
         detId[count]=rawdetId_;                                   
         float ped_ = (calibrations.pedestal(0)+calibrations.pedestal(1)+calibrations.pedestal(2)+calibrations.pedestal(3))/4;
	 float gain_= (calibrations.gain(0)+calibrations.gain(1)+calibrations.gain(2)+calibrations.gain(3))/4;
	
         rechit_calib = 0.075;

	 // do the HF digis
	 HFDataFrame frame(cell);
	 frame.setSize(1);

	 CaloSamples samples(cell, 1);
	 for (int j = 0; j <= 0x7F; j++) {
	   HcalQIESample adc(j);
	   frame.setSample(0,adc);
	   coder.adc2fC(frame,samples);
	   // if (phi_==1 && eta_==30) std::cout << "ADC = " << j << "; linearized = " << samples[0] << std::endl;
	   adc2fC_[j] = samples[0];
	   //    LUTwrite(count,j,std::min(std::max(0,int((adc2fC_[j] - ped_)*rechit_calib/lsb_/cosheta_[abs(eta_)])), 0x7F));
	  
	   // myvec.push_back(std::make_pair(rawdetId_,lin_lut_val[j]));           
	 }
         count++;
	 if (count != nluts) std::cout << "PROBLEM" << " " << count << std::endl;
       }

       /*
    HcalDetId id;
    for (int ieta=-42; ieta <= 42; ieta++) {
      for (int iphi = 0; iphi <= 72; iphi++) {
	for (int depth = 0; depth < 5; depth++) {
          for (int det = 1; det < 3; det++) {
	  id=HcalDetId((HcalSubdetector) det,ieta,iphi,depth);
	  if (theCaloTopo.valid(id)) {
	    for (iter2 = myvec.begin(); iter2 != myvec.end(); iter2++)
	      {
		if ((*iter2).first == id.rawId()) std::cout << "id: " << id.rawId() << " --> " << (*iter2).second << std::endl;
	      }
	  }

	  }
	}
      }
    }
       */

    
      
    	     
}



bool HcaluLUTTPGCoder::getadc2fCLUT() {
  char *filename = "CalibCalorimetry/HcalTPGAlgos/data/adc2fC.dat";
  std::ifstream userfile;
  userfile.open(filename);
  int tool;

  if( !userfile ) {
	  std::cout << "File " << filename << " with adc2fC LUT not found" << std::endl;
	  return false;
  }
  if (userfile) {
  userfile >> tool;

  if (tool != INPUT_LUT_SIZE) {
	  std::cout << "Wrong adc2fC LUT size: " << tool << " (expect 128)" << std::endl;
	return false;
  }
  for (int j=0; j<INPUT_LUT_SIZE; j++) {
	  userfile >> adc2fCLUT_[j]; // Read the ADC to fC LUT
	  // std::cout << adc2fCLUT_[j] << std::endl;
  }
  std::cout << "(1)Finished reading adc2fCLUT" << std::endl;

  userfile.close();
  std::cout << "(3)Finished reading adc2fCLUT" << std::endl;
  return true;
  }
  else {
    std::cout << "Problem!" << std::endl;
    return false;
  }
}


bool HcaluLUTTPGCoder::getped() {
	ped_ = 4.23729;
        ped_HF = 1.5625;
	return true;
}

bool HcaluLUTTPGCoder::getgain() {
	gain_ = 0.075;
	return true;
}

void HcaluLUTTPGCoder::generateILUTs() {
  if (!getadc2fCLUT()) throw cms::Exception("Missing/corrupted adc2fC LUT file");
  std::cout << "adc to fC LUT loaded..." << std::endl;
  
  if (!getped()) throw cms::Exception("Missing ped value");
  std::cout << "Pedestal = " << ped_ << " and HF:" << ped_HF << std::endl;
  if (!getgain()) throw cms::Exception("Missing gain value");
  std::cout << "Gain = " << gain_ << std::endl;
  
  //std::cout << adc2fCLUT_[0][127] << std::endl;
  char *filename = "test.dat";  
  std::ofstream myuserfile;
  myuserfile.open(filename, std::ofstream::out);
  std::cout << "File " << filename << " has been opened..." << std::endl;
  
  myuserfile << "29" << std::endl;
      
  myuserfile << "1	21	27	29	30	31	32	33	34	35	36	37	38	39	40	41	42	43	44	45	46	47	48	49	50	51	52	53	54" << std::endl;
  myuserfile << "20	26	28	29	30	31	32	33	34	35	36	37	38	39	40	41	42	43	44	45	46	47	48	49	50	51	52	53	54" << std::endl;

    
  //for (int i = 0; i < INPUT_LUT_SIZE; i++) {
    //std::cout << adc2fCLUT_[i] << std::endl;
 // }
  

  float cosheta[13], lsb = 1./16.;
  for (int i = 0; i < 13; i++) {
    cosheta[i] = cosh((theHFEtaBounds[i+1] + theHFEtaBounds[i])/2.);
  }
  for (int i = 0; i < INPUT_LUT_SIZE; i++) {
    
	  myuserfile << std::max(0,int((adc2fCLUT_[i] - ped_)/1.)) << " " << std::max(0,int((adc2fCLUT_[i] - ped_)/2.)) << " " << std::max(0,int((adc2fCLUT_[i] - ped_)/5.)) << " ";

	  for (int j = 0; j < 13; j++) {
	    if (j < 3) myuserfile << std::max(0,int((adc2fCLUT_[i]*2.6 - ped_HF)*gain_*0.90/lsb/cosheta[j])) << " ";
	    else if (j < 9)  myuserfile << std::max(0,int((adc2fCLUT_[i]*2.6 - ped_HF)*gain_/lsb/cosheta[j])) << " ";
	    else  myuserfile << std::max(0,int((adc2fCLUT_[i]*2.6 - ped_HF)*gain_*0.95/lsb/cosheta[j])) << " ";
	   
	  }
	  for (int j = 0; j < 13; j++) {
		if (j < 3)  myuserfile << std::max(0,int((adc2fCLUT_[i]*2.6 - ped_HF)*gain_*0.90/lsb/cosheta[j]));
                else if (j < 9)  myuserfile << std::max(0,int((adc2fCLUT_[i]*2.6 - ped_HF)*gain_/lsb/cosheta[j]));
                else  myuserfile << std::max(0,int((adc2fCLUT_[i]*2.6 - ped_HF)*gain_*0.95/lsb/cosheta[j]));
		  if (j < 12) myuserfile << " ";
		  else myuserfile << std::endl;
	  }
	  
  }

   myuserfile.close();
  //   std::cout << "File created and closed" << std::endl;

}

void HcaluLUTTPGCoder::loadILUTs(const char* filename) {
  int tool;
  std::ifstream userfile;
  userfile.open(filename);
  std::cout << filename << std::endl;
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
        
    //    std::cout << nluts << std::endl;
    /*
    for(int i = 0; i <nluts; i++) {
      for (int j=0; j<INPUT_LUT_SIZE; j++) { 
	std::cout << i << "[" << j << "] = "<< inputluts_[i][j] << std::endl;
      }
    }
    */

    // map |ieta| to LUT
    for (int j=1; j<=54; j++) {
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
  //const LUTType* lut=ietaILutMap_[df.id().ietaAbs()-1];
  const LUTType* lut=ietaILutMap_[df.id().ietaAbs()+13*(df.id().depth() - 1) - 1];

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
    
  }    

}

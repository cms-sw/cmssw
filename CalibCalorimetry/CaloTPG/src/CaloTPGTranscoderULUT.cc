#include "CalibCalorimetry/CaloTPG/src/CaloTPGTranscoderULUT.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/HcalTowerAlgo/interface/HcalTrigTowerGeometry.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include <iostream>
#include <fstream>

using namespace std;

HcalTrigTowerGeometry theTrigTowerGeometry;

CaloTPGTranscoderULUT::CaloTPGTranscoderULUT(const std::string& hcalFile) : 
  hcalITower_(N_TOWER,(const LUTType*)0)
{
  loadHCAL(hcalFile);
  loadhcalUncompress();

}

void CaloTPGTranscoderULUT::loadHCAL(const std::string& filename) {
  static const int LUT_SIZE=1024;
  int tool;
  std::ifstream userfile;
  userfile.open(filename.c_str());

  if( userfile ) {
    int nluts;
    std::vector<int> loieta,hiieta;
    userfile >> nluts;

    hcal_.resize(nluts);
    for (int i=0; i<nluts; i++) {
      hcal_[i].resize(LUT_SIZE); 
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
	  userfile >> hcal_[i][j];}
    }
    
    userfile.close();
    /*    
    std::cout << nluts << std::endl;

    for(int i = 0; i <nluts; i++) {
      for (int j=0; j<LUT_SIZE; j++) { 
	std::cout << i << "[" << j << "] = "<< hcal_[i][j] << std::endl;
      }
    }
    */

    // map |ieta| to LUT
    for (int j=1; j<=N_TOWER; j++) {
      int ilut=-1;
      for (ilut=0; ilut<nluts; ilut++)
	if (j>=loieta[ilut] && j<=hiieta[ilut]) break;
      if (ilut==nluts) {
	hcalITower_[j-1]=0;
	// TODO: log warning
      }
      else hcalITower_[j-1]=&(hcal_[ilut]);
      //      std::cout << j << "->" << ilut << std::endl;
    }    
  } else {
    throw cms::Exception("Invalid Data") << "Unable to read " << filename;
  }
}

void CaloTPGTranscoderULUT::loadhcalUncompress() {
  std::ifstream userfile;
  userfile.open("CalibCalorimetry/CaloTPG/data/TPGcalcDecompress2.txt");
  static const int etabound = 32;
  static const int tpgmax = 255;
  if( userfile ) 
    {
     for (int i=0; i<etabound; i++) { 
      for(int j = 0; j <tpgmax; j++) {
	  userfile >> hcaluncomp_[i][j];}
    }
     //  cout<<"test hcal"<<endl;
    userfile.close();

    }
  else {
    throw cms::Exception("Invalid Data") << "Unable to read uncompress file";
  }
}



HcalTriggerPrimitiveSample CaloTPGTranscoderULUT::hcalCompress(const HcalTrigTowerDetId& id, unsigned int sample, bool fineGrain) const {
  int itower=id.ietaAbs();
  const LUTType* lut=hcalITower_[itower-1];
  if (lut==0) {
    throw cms::Exception("Invalid Data") << "No LUT available for " << itower;
  } 
  if (sample>=lut->size()) {
    throw cms::Exception("Out of Range") << "LUT has " << lut->size() << " entries for " << itower << " but " << sample << " was requested.";
  }
  //  std::cout << id << ":" << sample << "-->" << (*lut)[sample] << std::endl;
  return HcalTriggerPrimitiveSample((*lut)[sample],fineGrain,0,0);
}


double CaloTPGTranscoderULUT::hcaletValue(const int& ieta, const int& compET){
  
  double etvalue = hcaluncomp_[ieta][compET];//*cos(eta_ave);
  return(etvalue);
}


EcalTriggerPrimitiveSample CaloTPGTranscoderULUT::ecalCompress(const EcalTrigTowerDetId& id, unsigned int sample, bool fineGrain) const {
  throw cms::Exception("Not Implemented") << "CaloTPGTranscoderULUT::ecalCompress";
}

void CaloTPGTranscoderULUT::rctEGammaUncompress(const HcalTrigTowerDetId& hid, const HcalTriggerPrimitiveSample& hc,
						const EcalTrigTowerDetId& eid, const EcalTriggerPrimitiveSample& ec, 
						unsigned int& et, bool& egVecto, bool& activity) const {
  throw cms::Exception("Not Implemented") << "CaloTPGTranscoderULUT::rctEGammaUncompress";
}
void CaloTPGTranscoderULUT::rctJetUncompress(const HcalTrigTowerDetId& hid, const HcalTriggerPrimitiveSample& hc,
					     const EcalTrigTowerDetId& eid, const EcalTriggerPrimitiveSample& ec, 
					     unsigned int& et) const {
  throw cms::Exception("Not Implemented") << "CaloTPGTranscoderULUT::rctJetUncompress";
}

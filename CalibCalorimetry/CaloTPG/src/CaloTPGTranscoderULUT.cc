#include "CalibCalorimetry/CaloTPG/src/CaloTPGTranscoderULUT.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/HcalTowerAlgo/interface/HcalTrigTowerGeometry.h"
#include <iostream>
#include <fstream>

using namespace std;

HcalTrigTowerGeometry theTrigTowerGeometry;

CaloTPGTranscoderULUT::CaloTPGTranscoderULUT(const std::string& hcalFile1,
					     const std::string& hcalFile2) : 
  hcalITower_(N_TOWER,(const LUTType*)0)
{
  loadHCAL(hcalFile1);
  loadhcalUncompress(hcalFile2);
}

CaloTPGTranscoderULUT::~CaloTPGTranscoderULUT() {
  for (int i = 0; i < noutluts; i++) {
    if (outputLUT[i] != 0) delete [] outputLUT[i];
  }
}


void CaloTPGTranscoderULUT::loadHCAL(const std::string& filename) {
  int tool;
  std::ifstream userfile;

  // std::cout << "Initializing compression LUT's" << std::endl;
  for (int i = 0; i < noutluts; i++) outputLUT[i] = 0;
  int maxid = 0, minid = 0x7FFFFFFF, rawid = 0;
  for (int ieta=-32; ieta <= 32; ieta++) {
    for (int iphi = 1; iphi <= 72; iphi++) {
		if (HTvalid(ieta,iphi)) {
          rawid = GetOutputLUTId(ieta, iphi);
          if (outputLUT[rawid] != 0) std::cout << "Error: LUT with (ieta,iphi) = (" << ieta << "," << iphi << ") has been previously allocated!" << std::endl;
          else outputLUT[rawid] = new LUT[OUTPUT_LUT_SIZE];
          if (rawid < minid) minid = rawid;
          if (rawid > maxid) maxid = rawid;
        }
	}
  }

  userfile.open((char *)filename.data());
  //std::cout << filename << std::endl;
  if( userfile ) {
    int nluts = 0;
	std::string s;
    std::vector<int> loieta,hiieta;
    std::vector<int> loiphi,hiiphi;
    getline(userfile,s);
    //std::cout << "Reading Compression LUT's for: " << s << std::endl;
    getline(userfile,s);
//
	unsigned int index = 0;
	while (index < s.length()) {
	  while (isspace(s[index])) index++;
	  if (index < s.length()) nluts++;
	  while (!isspace(s[index])) index++;
	}
	//std::cout << "Found " << nluts << " LUTs" << std::endl;
	for (unsigned int i=0; i<=s.length(); i++) userfile.unget(); //rewind last line

    outputluts_.resize(nluts);
    for (int i=0; i<nluts; i++) {
      outputluts_[i].resize(OUTPUT_LUT_SIZE); 
    }
    
    //std::cout << "EtaMin = ";
    for (int i=0; i<nluts; i++) {
      userfile >> tool;
      loieta.push_back(tool);
      //  std::cout << tool << " ";
    }
    //std::cout << std::endl << "EtaMax = ";
    for (int i=0; i<nluts; i++) {
      userfile >> tool;
      hiieta.push_back(tool);
      //  std::cout << tool << " ";
    }
    //std::cout << std::endl << "PhiMin = ";
    for (int i=0; i<nluts; i++) {
      userfile >> tool;
      loiphi.push_back(tool);
      //  std::cout << tool << " ";
    }
    //std::cout << std::endl << "PhiMax = ";
    for (int i=0; i<nluts; i++) {
      userfile >> tool;
      hiiphi.push_back(tool);
      //  std::cout << tool << " ";
    }
    //std::cout << std::endl;
	
    for (unsigned int j=0; j<OUTPUT_LUT_SIZE; j++) { 
      for(int i=0; i <nluts; i++) {
		userfile >> tool;
		if (tool < 0) {
			std::cout << "Error: LUT can't have negative numbers; 0 used instead: " << i << ", " << j << " : = " << tool << std::endl;
			tool = 0;
		} else if (tool > 0xff) {
			std::cout << "Error: LUT can't have >8-bit numbers; 0xff used instead: " << i << ", " << j << " : = " << tool << std::endl;
			tool = 0xff;
		}
		outputluts_[i][j] = tool;
		if (userfile.eof()) std::cout << "Error: LUT file is truncated or has a wrong format: " << i << "," << j << std::endl;
	  }
    }
    userfile.close();
	
	HcalDetId cell;
	int id, ntot = 0;
	for (int i=0; i < nluts; i++) {
		int nini = 0;
     		for (int iphi = loiphi[i]; iphi <= hiiphi[i]; iphi++) {      
       			for (int ieta=loieta[i]; ieta <= hiieta[i]; ieta++) {
	 				if (HTvalid(ieta,iphi)) {  
	   					id = GetOutputLUTId(ieta,iphi);
	   					if (outputLUT[id] == 0) throw cms::Exception("PROBLEM: outputLUT has not been initialized for ieta, iphi, id = ") << ieta << ", " << iphi << ", " << id << std::endl;
		    			for (int j = 0; j <= 0x3ff; j++) outputLUT[id][j] = outputluts_[i][j];
						nini++;
						ntot++;
	 				}
       			}
       }
		//std::cout << nini << " LUT's have been initialized for eta = [" << loieta[i] << "," << hiieta[i] << "]; iphi = [" << loiphi[i] << "," << hiiphi[i] << "]"  << std::endl;
    }
	//std::cout << "Total of " << ntot << " comression LUT's have been initialized" << std::endl;
  } 

/*
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
  } */
}

void CaloTPGTranscoderULUT::loadhcalUncompress(const std::string& filename) {
  std::ifstream userfile;
  userfile.open(filename.c_str());
  static const int etabound = 32;
  static const int tpgmax = 256;
  if( userfile ) 
    {
     for (int i=0; i<tpgmax; i++) { 
      for(int j = 1; j <=etabound; j++) {
	 userfile >> hcaluncomp_[j][i];}
    }
     //  cout<<"test hcal"<<endl;
    userfile.close();

    }
  else {
    throw cms::Exception("Invalid Data") << "Unable to read uncompress file" << filename;
  }
}

HcalTriggerPrimitiveSample CaloTPGTranscoderULUT::hcalCompress(const HcalTrigTowerDetId& id, unsigned int sample, bool fineGrain) const {
  int ieta = id.ieta();
  int iphi = id.iphi();
  if (abs(ieta) > 28) iphi = iphi/4 + 1; // Changing iphi index from 1, 5, ..., 69 to 1, 2, ..., 18
  int itower = GetOutputLUTId(ieta,iphi);
//  std::cout << "Compressing ieta, iphi, tower: " << ieta << ", " << iphi << ", " << itower << std::endl;
  if (itower < 0) cms::Exception("Invalid Data") << "No trigger tower found for ieta, iphi = " << ieta << ", " << iphi;
  if (sample >= OUTPUT_LUT_SIZE) {
//    std::cout << "Out of range entry in the LUT: " << sample << std::endl;
    throw cms::Exception("Out of Range") << "LUT has 1024 entries for " << itower << " but " << sample << " was requested.";
    sample=OUTPUT_LUT_SIZE - 1;
  }
  //  std::cout << id << ":" << sample << "-->" << (*lut)[sample] << std::endl;
  return HcalTriggerPrimitiveSample(outputLUT[itower][sample],fineGrain,0,0);
}


double CaloTPGTranscoderULUT::hcaletValue(const int& ieta, const int& compET) const {
  double etvalue = hcaluncomp_[ieta][compET];//*cos(eta_ave);
  return(etvalue);
}

double CaloTPGTranscoderULUT::hcaletValue(const HcalTrigTowerDetId& hid, const HcalTriggerPrimitiveSample& hc) const {
  int ieta = hid.ietaAbs();
  int compET = hc.compressedEt();
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

bool CaloTPGTranscoderULUT::HTvalid(const int ieta, const int iphi) const {
	if (iphi <= 0 || ieta == 0) return false;
	if (abs(ieta) > 32) return false;
	else if (abs(ieta) > 28 && iphi > 18) return false;
	else if (iphi > 72) return false;
	return true;
}

int CaloTPGTranscoderULUT::GetOutputLUTId(const int ieta, const int iphi) const {
	if (HTvalid(ieta, iphi)) {
		int offset = 0, ietaabs;
		ietaabs = abs(ieta);
		if (ieta < 0) offset = noutluts/2;
		if (ietaabs < 29) return 72*(ietaabs - 1) + (iphi - 1) + offset;
		else return 18*(ietaabs - 29) + iphi + 2015 + offset;
	} else return -1;	
}

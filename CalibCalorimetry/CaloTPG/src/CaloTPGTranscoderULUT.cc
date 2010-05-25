#include "CalibCalorimetry/CaloTPG/src/CaloTPGTranscoderULUT.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/HcalTowerAlgo/interface/HcalTrigTowerGeometry.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <iostream>
#include <fstream>
#include <math.h>

using namespace std;

HcalTrigTowerGeometry theTrigTowerGeometry;

CaloTPGTranscoderULUT::CaloTPGTranscoderULUT() {
  throw cms::Exception("CaloTPGTranscoderULUT") << "This constructor has been deprecated. Please modifilied your code\n";
  DecompressionFile = "";
  loadHCALCompress();
}

CaloTPGTranscoderULUT::CaloTPGTranscoderULUT(const std::string& hcalFile1, const std::string& hcalFile2){
  throw cms::Exception("CaloTPGTranscoderULUT") << "This constructor has been deprecated. Please modifilied your code\n";
   if (hcalFile1.empty() && hcalFile2.empty()) {
      DecompressionFile = "";
      loadHCALCompress();
   }
   else {
      DecompressionFile = hcalFile2;
      loadHCALCompress(hcalFile1);
   }
}

//New constructor
CaloTPGTranscoderULUT::CaloTPGTranscoderULUT(const std::vector<int>& _ietal,const std::vector<int>& _ietah,const std::vector<int>& _zs,const std::vector<int>& _lutfactor, const double& _rctlsb, const double& _nominalgain, const std::string& hcalFile1="", const std::string& hcalFile2=""){

   setLUTGranularity(_ietal, _ietah, _zs, _lutfactor);
   setRCTLSB(_rctlsb);
   setNominalGain(_nominalgain);

   if (hcalFile1.empty() && hcalFile2.empty()) {
      DecompressionFile = "";
      loadHCALCompress();
   }
   else {
      DecompressionFile = hcalFile2;
      loadHCALCompress(hcalFile1);
   }
}

CaloTPGTranscoderULUT::~CaloTPGTranscoderULUT() {
  for (int i = 0; i < NOUTLUTS; i++) {
    if (outputLUT[i] != 0) delete [] outputLUT[i];
  }
}

void CaloTPGTranscoderULUT::loadHCALCompress() {
// Initialize analytical compression LUT's here
  if (OUTPUT_LUT_SIZE != (unsigned int) 0x400) std::cout << "Error: Analytic compression expects 10-bit LUT; found LUT with " << OUTPUT_LUT_SIZE << " entries instead" << std::endl;
  //std::cout << "Built analytical (HB/HE) and identity (HF) compression LUTs" << std::endl;
  for (unsigned int i=0; i < OUTPUT_LUT_SIZE; i++) {
	AnalyticalLUT[i] = (unsigned int)(sqrt(14.94*log(1.+i/14.94)*i) + 0.5);
	IdentityLUT[i] = min(i,0xffu);
  }
 
  int nlut = 0;
  for (int i = 0; i < NOUTLUTS; i++) outputLUT[i] = 0;
  for (int ieta=-32; ieta <= 32; ieta++) {
	int ir = 0;
	while (ir < NR && abs(ieta) >= ietal[ir]) ir++;
	ir--;
	if (ir >= 0 && abs(ieta) <= ietah[ir]) {
		for (int iphi = 1; iphi <= 72; iphi++) {
			if (HTvalid(ieta,iphi)) {
				int rawid = GetOutputLUTId(ieta,iphi);
				if (outputLUT[rawid] != 0) std::cout << "Error: LUT with (ieta,iphi) = (" << ieta << "," << iphi << ") has been previously allocated!" << std::endl;
				else 
				{
					nlut++;
					outputLUT[rawid] = new LUT[OUTPUT_LUT_SIZE];
					for (int k=0; k<ZS[ir]; k++) outputLUT[rawid][k] = 0;
					for (unsigned int k=ZS[ir]; k < OUTPUT_LUT_SIZE; k++) outputLUT[rawid][k] = (abs(ieta) < theTrigTowerGeometry.firstHFTower()) ? AnalyticalLUT[k] : IdentityLUT[k];
				}
			}
		}
	}
  }
}

void CaloTPGTranscoderULUT::loadHCALCompress(const std::string& filename) {
  int tool;
  std::ifstream userfile;

  std::cout << "Initializing compression LUT's from " << (char *)filename.data() << std::endl;
  for (int i = 0; i < NOUTLUTS; i++) outputLUT[i] = 0;
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

  if( userfile ) {
    int nluts = 0;
	std::string s;
    std::vector<int> loieta,hiieta;
    std::vector<int> loiphi,hiiphi;
    getline(userfile,s);

    getline(userfile,s);

	unsigned int index = 0;
	while (index < s.length()) {
	  while (isspace(s[index])) index++;
	  if (index < s.length()) nluts++;
	  while (!isspace(s[index])) index++;
	}
	for (unsigned int i=0; i<=s.length(); i++) userfile.unget(); //rewind last line
    outputluts_.resize(nluts);
    for (int i=0; i<nluts; i++) outputluts_[i].resize(OUTPUT_LUT_SIZE);
    
    
    for (int i=0; i<nluts; i++) {
      userfile >> tool;
      loieta.push_back(tool);
    }
    
    for (int i=0; i<nluts; i++) {
      userfile >> tool;
      hiieta.push_back(tool);    
    }
   
    for (int i=0; i<nluts; i++) {
      userfile >> tool;
      loiphi.push_back(tool);
	
    }
    
    for (int i=0; i<nluts; i++) {
      userfile >> tool;
      hiiphi.push_back(tool);
    
    }
   	
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
	
    }
	
  } else {
    
	loadHCALCompress();
  }
}

void CaloTPGTranscoderULUT::loadHCALUncompress() const {
    hcaluncomp_.resize(NOUTLUTS);
    for (int i = 0; i < NOUTLUTS; i++) hcaluncomp_[i].resize(TPGMAX);
	unsigned int uncompress[TPGMAX];
	for (int j = -theTrigTowerGeometry.nTowers(); j <= theTrigTowerGeometry.nTowers(); j++) {
		double eta_low = 0., eta_high = 0., factor = 0.;
		//theTrigTowerGeometry.towerEtaBounds(abs(j),eta_low,eta_high); // Should use j, not abs(j), once the geometry bug is fixed!
		theTrigTowerGeometry.towerEtaBounds(j,eta_low,eta_high); 

		factor = nominal_gain/cosh((eta_low + eta_high)/2.);
		if (factor < 0) factor = -factor;
		int ir = 0;
		while (ir < NR && abs(j) >= ietal[ir]) ir++;
		ir--;
		if (ir >= 0 && abs(j) <= ietah[ir]) {
			factor *= LUTfactor[ir];
		}
		if (abs(j) >= theTrigTowerGeometry.firstHFTower()) factor = RCTLSB_factor;
		for (int iphi = 1; iphi <= 72; iphi++) {
			int itower = GetOutputLUTId(j,iphi);
			if (itower >= 0) {
				unsigned int compressed = 0, i = 0, low = 0, high = 255;
				for (int k = 0; k < 256; k++) uncompress[k] = 0;
// Now handle ZS - set all the uncompressed values below the zero suppression level to 0
				while (outputLUT[itower][i] == 0) i++;
				for (compressed = 0; compressed < i; compressed++) uncompress[compressed] = 0;
				while (i < OUTPUT_LUT_SIZE && compressed < 256) {
					if (outputLUT[itower][i] == compressed) {
						low = i;
						do i++;
						while (outputLUT[itower][i] == compressed);
						high = --i;
						if (compressed == 0 || high == OUTPUT_LUT_SIZE-1) uncompress[compressed++] = low;
						else uncompress[compressed++] = (low + high + 1)/2; // return the middle of the range
					} else i++;
				}
				for (unsigned int i=0; i < 256; i++) {
					if (outputLUT[itower][uncompress[i]] != i && outputLUT[itower][uncompress[i]] != 0) std::cout << "Bad decompression: itower(ieta, iphi), i = " << itower << "(" << j << ", " << iphi << "), " << i << "; uncompressed: " << uncompress[i] << ", comp[uncomp] = " << int(outputLUT[itower][uncompress[i]]) << std::endl;
				}
				for (int i=0; i < 256; i++) hcaluncomp_[itower][i] = uncompress[i]*factor;
			}
		}
	}

}

void CaloTPGTranscoderULUT::loadHCALUncompress(const std::string& filename) const {
  std::ifstream userfile;
  userfile.open(filename.c_str());
  
  hcaluncomp_.resize(NOUTLUTS);
  for (int i = 0; i < NOUTLUTS; i++) hcaluncomp_[i].resize(TPGMAX);
  
  static const int etabound = 32;
  if( userfile ) {
	 double et;
     for (int i=0; i<TPGMAX; i++) { 
      for(int j = 1; j <=etabound; j++) {
	    userfile >> et;
		for (int iphi = 1; iphi <= 72; iphi++) {
		  int itower = GetOutputLUTId(j,iphi);
		  if (itower >= 0) hcaluncomp_[itower][i] = et;
		  itower = GetOutputLUTId(-j,iphi);
		  if (itower >= 0) hcaluncomp_[itower][i] = et;		  
		}
	  }
	 }
	 userfile.close();
  }
  else {

	loadHCALUncompress();
  }
}

HcalTriggerPrimitiveSample CaloTPGTranscoderULUT::hcalCompress(const HcalTrigTowerDetId& id, unsigned int sample, bool fineGrain) const {
  int ieta = id.ieta();
  int iphi = id.iphi();
//  if (abs(ieta) > 28) iphi = iphi/4 + 1; // Changing iphi index from 1, 5, ..., 69 to 1, 2, ..., 18
  int itower = GetOutputLUTId(ieta,iphi);

  if (itower < 0) cms::Exception("Invalid Data") << "No trigger tower found for ieta, iphi = " << ieta << ", " << iphi;
  if (sample >= OUTPUT_LUT_SIZE) {

    throw cms::Exception("Out of Range") << "LUT has 1024 entries for " << itower << " but " << sample << " was requested.";
    sample=OUTPUT_LUT_SIZE - 1;
  }

  return HcalTriggerPrimitiveSample(outputLUT[itower][sample],fineGrain,0,0);
}

double CaloTPGTranscoderULUT::hcaletValue(const int& ieta, const int& iphi, const int& compET) const {
  if (hcaluncomp_.empty()) {

	CaloTPGTranscoderULUT::loadHCALUncompress(DecompressionFile);
  }
  double etvalue = 0.;
  int itower = GetOutputLUTId(ieta,iphi);
  if (itower < 0) std::cout << "hcaletValue error: no decompression LUT found for ieta, iphi = " << ieta << ", " << iphi << std::endl;
  else if (compET < 0 || compET > 0xff) std::cout << "hcaletValue error: compressed value out of range: eta, phi, cET = " << ieta << ", " << iphi << ", " << compET << std::endl;
  else etvalue = hcaluncomp_[itower][compET];
  return(etvalue);
}

double CaloTPGTranscoderULUT::hcaletValue(const int& ieta, const int& compET) const {
// This is now an obsolete method; we return the AVERAGE over all the allowed iphi channels if it's invoked
// The user is encouraged to use hcaletValue(const int& ieta, const int& iphi, const int& compET) instead

  if (hcaluncomp_.empty()) {
	std::cout << "Initializing the RCT decompression table from the file: " << DecompressionFile << std::endl;
	CaloTPGTranscoderULUT::loadHCALUncompress(DecompressionFile);
  }

  double etvalue = 0.;
  if (compET < 0 || compET > 0xff) std::cout << "hcaletValue error: compressed value out of range: eta, cET = " << ieta << ", " << compET << std::endl;
  else {
	int nphi = 0;
	for (int iphi=1; iphi <= 72; iphi++) {
		if (HTvalid(ieta,iphi)) {
			nphi++;
			int itower = GetOutputLUTId(ieta,iphi);
			etvalue += hcaluncomp_[itower][compET];
		}
	}
	if (nphi > 0) etvalue /= nphi;
	else std::cout << "hcaletValue error: no decompression LUTs found for any iphi for ieta = " << ieta << std::endl;
  }
  return(etvalue);
}

double CaloTPGTranscoderULUT::hcaletValue(const HcalTrigTowerDetId& hid, const HcalTriggerPrimitiveSample& hc) const {
  if (hcaluncomp_.empty()) loadHCALUncompress(DecompressionFile);

  int ieta = hid.ieta();			// No need to check the validity,
  int iphi = hid.iphi();			// as the values are guaranteed
  int compET = hc.compressedEt();	// to be within the range by the class
  int itower = GetOutputLUTId(ieta,iphi);
  double etvalue = hcaluncomp_[itower][compET];
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

bool CaloTPGTranscoderULUT::HTvalid(const int ieta, const int iphiin) const {
	int iphi = iphiin;
	if (iphi <= 0 || iphi > 72 || ieta == 0 || abs(ieta) > 32) return false;
	if (abs(ieta) > 28) {
	  if (newHFphi) {
	    if ((iphi/4)*4 + 1 != iphi) return false;
	    iphi = iphi/4 + 1;
	  }
	  if (iphi > 18) return false;
	}
	return true;
}

int CaloTPGTranscoderULUT::GetOutputLUTId(const int ieta, const int iphiin) const {
	int iphi = iphiin;
	if (HTvalid(ieta, iphi)) {
		int offset = 0, ietaabs;
		ietaabs = abs(ieta);
		if (ieta < 0) offset = NOUTLUTS/2;
		if (ietaabs < 29) return 72*(ietaabs - 1) + (iphi - 1) + offset;
		else {
		  if (newHFphi) iphi = iphi/4 + 1;
		  return 18*(ietaabs - 29) + iphi + 2015 + offset;
		}
	} else return -1;	
}

std::vector<unsigned char> CaloTPGTranscoderULUT::getCompressionLUT(HcalTrigTowerDetId id) const {
   std::vector<unsigned char> lut;
   int itower = GetOutputLUTId(id.ieta(),id.iphi());
   if (itower >= 0) {
         lut.resize(OUTPUT_LUT_SIZE);
	 for (unsigned int i = 0; i < OUTPUT_LUT_SIZE; i++) lut[i]=outputLUT[itower][i];
   } 
   return lut;
}

void CaloTPGTranscoderULUT::setLUTGranularity(const std::vector<int>& _ietal,const std::vector<int>& _ietah,const std::vector<int>& _zs,const std::vector<int>& _lutfactor){
   ietal = _ietal;
   ietah = _ietah;
   ZS = _zs;
   LUTfactor = _lutfactor;

   NR = ietal.size();
   if (ietah.size() != NR || ZS.size() != NR || LUTfactor.size() != NR)
      throw cms::Exception("LUT Granularity") << "Different size of LUT granularity.\n";
}

void CaloTPGTranscoderULUT::setRCTLSB(const double& _rctlsb){
   RCTLSB = _rctlsb;
   if (RCTLSB != 0.25 && RCTLSB != 0.5)
      throw cms::Exception("RCTLSB") << "RCTLSB must be 0.25 or 0.5.\n";

   if (RCTLSB == 0.25) RCTLSB_factor = 1/4.;
   else if (RCTLSB == 0.5) RCTLSB_factor = 1/8.;
}

void CaloTPGTranscoderULUT::setNominalGain(const double& _nominal_gain){
   nominal_gain = _nominal_gain;
}

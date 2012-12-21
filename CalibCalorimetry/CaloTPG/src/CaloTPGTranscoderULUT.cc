#include "CalibCalorimetry/CaloTPG/src/CaloTPGTranscoderULUT.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/HcalTowerAlgo/interface/HcalTrigTowerGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <iostream>
#include <fstream>
#include <math.h>

//#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CondFormats/DataRecord/interface/HcalLutMetadataRcd.h"

using namespace std;


CaloTPGTranscoderULUT::CaloTPGTranscoderULUT(const std::string& compressionFile,
                                             const std::string& decompressionFile)
                                                : isLoaded_(false), nominal_gain_(0.), rctlsb_factor_(0.),
                                                  compressionFile_(compressionFile),
                                                  decompressionFile_(decompressionFile)
{
  for (int i = 0; i < NOUTLUTS; i++) outputLUT_[i] = 0;
}

CaloTPGTranscoderULUT::~CaloTPGTranscoderULUT() {
  for (int i = 0; i < NOUTLUTS; i++) {
    if (outputLUT_[i] != 0) delete [] outputLUT_[i];
  }
}

void CaloTPGTranscoderULUT::loadHCALCompress() const{
// Initialize analytical compression LUT's here
   // TODO cms::Error log
  if (OUTPUT_LUT_SIZE != (unsigned int) 0x400) std::cout << "Error: Analytic compression expects 10-bit LUT; found LUT with " << OUTPUT_LUT_SIZE << " entries instead" << std::endl;

  std::vector<unsigned int> analyticalLUT(OUTPUT_LUT_SIZE, 0);
  std::vector<unsigned int> identityLUT(OUTPUT_LUT_SIZE, 0);

  // Compute compression LUT
  for (unsigned int i=0; i < OUTPUT_LUT_SIZE; i++) {
	analyticalLUT[i] = (unsigned int)(sqrt(14.94*log(1.+i/14.94)*i) + 0.5);
	identityLUT[i] = min(i,0xffu);
  }
 
  for (int ieta=-32; ieta <= 32; ieta++){
     for (int iphi = 1; iphi <= 72; iphi++){
        if (!HTvalid(ieta,iphi)) continue;
        int lutId = getOutputLUTId(ieta,iphi);
        // TODO cms::Error log
        if (outputLUT_[lutId] != 0){
           std::cout << "Error: LUT with (ieta,iphi) = (" << ieta << "," << iphi << ") has been previously allocated!" << std::endl;
           continue;
        }

        outputLUT_[lutId] = new LUT[OUTPUT_LUT_SIZE];

        HcalTrigTowerDetId id(ieta, iphi);
        const HcalLutMetadatum *meta = lutMetadata_->getValues(id);
        int threshold = meta->getOutputLutThreshold();

        for (int i = 0; i < threshold; ++i)
           outputLUT_[lutId][i] = 0;

        for (unsigned int i = threshold; i < OUTPUT_LUT_SIZE; ++i)
           outputLUT_[lutId][i] = (abs(ieta) < theTrigTowerGeometry->firstHFTower()) ? analyticalLUT[i] : identityLUT[i];
     } //for iphi
  } //for ieta
}

void CaloTPGTranscoderULUT::loadHCALCompress(const std::string& filename) const{
  int tool;
  std::ifstream userfile;
  std::vector< std::vector<LUT> > outputluts;

  std::cout << "Initializing compression LUT's from " << (char *)filename.data() << std::endl;
  for (int i = 0; i < NOUTLUTS; i++) outputLUT_[i] = 0;
  int maxid = 0, minid = 0x7FFFFFFF, rawid = 0;
  for (int ieta=-32; ieta <= 32; ieta++) {
    for (int iphi = 1; iphi <= 72; iphi++) {
		if (HTvalid(ieta,iphi)) {
          rawid = getOutputLUTId(ieta, iphi);
          if (outputLUT_[rawid] != 0) std::cout << "Error: LUT with (ieta,iphi) = (" << ieta << "," << iphi << ") has been previously allocated!" << std::endl;
          else outputLUT_[rawid] = new LUT[OUTPUT_LUT_SIZE];
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
    outputluts.resize(nluts);
    for (int i=0; i<nluts; i++) outputluts[i].resize(OUTPUT_LUT_SIZE);
    
    
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
		outputluts[i][j] = tool;
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
	   					id = getOutputLUTId(ieta,iphi);
	   					if (outputLUT_[id] == 0) throw cms::Exception("PROBLEM: outputLUT_ has not been initialized for ieta, iphi, id = ") << ieta << ", " << iphi << ", " << id << std::endl;
		    			for (int j = 0; j <= 0x3ff; j++) outputLUT_[id][j] = outputluts[i][j];
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
   hcaluncomp_.clear();
   for (int i = 0; i < NOUTLUTS; i++){
      RCTdecompression decompressionTable(TPGMAX,0);
      hcaluncomp_.push_back(decompressionTable);
   }

   for (int ieta = -32; ieta <= 32; ++ieta){

      double eta_low = 0., eta_high = 0.;
		theTrigTowerGeometry->towerEtaBounds(ieta,eta_low,eta_high); 
      double cosh_ieta = fabs(cosh((eta_low + eta_high)/2.));

		for (int iphi = 1; iphi <= 72; iphi++) {
         if (!HTvalid(ieta, iphi)) continue;

			int lutId = getOutputLUTId(ieta,iphi);
         HcalTrigTowerDetId id(ieta, iphi);

         const HcalLutMetadatum *meta = lutMetadata_->getValues(id);
         double factor = 0.;

         // HF
         if (abs(ieta) >= theTrigTowerGeometry->firstHFTower())
            factor = rctlsb_factor_;
         // HBHE
         else 
            factor = nominal_gain_ / cosh_ieta * meta->getLutGranularity();

         // tpg - compressed value
         unsigned int tpg = outputLUT_[lutId][0];
         int low = 0;
         for (unsigned int i = 0; i < OUTPUT_LUT_SIZE; ++i){
            if (outputLUT_[lutId][i] != tpg){
               unsigned int mid = (low + i)/2;
               hcaluncomp_[lutId][tpg] = (tpg == 0 ? low : factor * mid);
               low = i;
               tpg = outputLUT_[lutId][i];
            }
         }
         hcaluncomp_[lutId][tpg] = factor * low;
		} // for phi
	} // for ieta
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
		  int itower = getOutputLUTId(j,iphi);
		  if (itower >= 0) hcaluncomp_[itower][i] = et;
		  itower = getOutputLUTId(-j,iphi);
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
  int itower = getOutputLUTId(ieta,iphi);

  if (itower < 0) throw cms::Exception("Invalid Data") << "No trigger tower found for ieta, iphi = " << ieta << ", " << iphi;
  if (sample >= OUTPUT_LUT_SIZE) {

    throw cms::Exception("Out of Range") << "LUT has 1024 entries for " << itower << " but " << sample << " was requested.";
    sample=OUTPUT_LUT_SIZE - 1;
  }

  return HcalTriggerPrimitiveSample(outputLUT_[itower][sample],fineGrain,0,0);
}

double CaloTPGTranscoderULUT::hcaletValue(const int& ieta, const int& iphi, const int& compET) const {
  if (hcaluncomp_.empty()) {

	CaloTPGTranscoderULUT::loadHCALUncompress(decompressionFile_);
  }
  double etvalue = 0.;
  int itower = getOutputLUTId(ieta,iphi);
  if (itower < 0) std::cout << "hcaletValue error: no decompression LUT found for ieta, iphi = " << ieta << ", " << iphi << std::endl;
  else if (compET < 0 || compET > 0xff) std::cout << "hcaletValue error: compressed value out of range: eta, phi, cET = " << ieta << ", " << iphi << ", " << compET << std::endl;
  else etvalue = hcaluncomp_[itower][compET];
  return(etvalue);
}

double CaloTPGTranscoderULUT::hcaletValue(const int& ieta, const int& compET) const {
// This is now an obsolete method; we return the AVERAGE over all the allowed iphi channels if it's invoked
// The user is encouraged to use hcaletValue(const int& ieta, const int& iphi, const int& compET) instead

  if (hcaluncomp_.empty()) {
	std::cout << "Initializing the RCT decompression table from the file: " << decompressionFile_ << std::endl;
	CaloTPGTranscoderULUT::loadHCALUncompress(decompressionFile_);
  }

  double etvalue = 0.;
  if (compET < 0 || compET > 0xff) std::cout << "hcaletValue error: compressed value out of range: eta, cET = " << ieta << ", " << compET << std::endl;
  else {
	int nphi = 0;
	for (int iphi=1; iphi <= 72; iphi++) {
		if (HTvalid(ieta,iphi)) {
			nphi++;
			int itower = getOutputLUTId(ieta,iphi);
			etvalue += hcaluncomp_[itower][compET];
		}
	}
	if (nphi > 0) etvalue /= nphi;
	else std::cout << "hcaletValue error: no decompression LUTs found for any iphi for ieta = " << ieta << std::endl;
  }
  return(etvalue);
}

double CaloTPGTranscoderULUT::hcaletValue(const HcalTrigTowerDetId& hid, const HcalTriggerPrimitiveSample& hc) const {
  if (hcaluncomp_.empty()) loadHCALUncompress(decompressionFile_);

  int ieta = hid.ieta();			// No need to check the validity,
  int iphi = hid.iphi();			// as the values are guaranteed
  int compET = hc.compressedEt();	// to be within the range by the class
  int itower = getOutputLUTId(ieta,iphi);
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

int CaloTPGTranscoderULUT::getOutputLUTId(const int ieta, const int iphiin) const {
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
   int itower = getOutputLUTId(id.ieta(),id.iphi());
   if (itower >= 0) {
         lut.resize(OUTPUT_LUT_SIZE);
	 for (unsigned int i = 0; i < OUTPUT_LUT_SIZE; i++) lut[i]=outputLUT_[itower][i];
   } 
   return lut;
}

void CaloTPGTranscoderULUT::setup(const edm::EventSetup& es, Mode mode=All) const{
   if (isLoaded_) return;
   // TODO Try/except
   es.get<HcalLutMetadataRcd>().get(lutMetadata_);
   es.get<CaloGeometryRecord>().get(theTrigTowerGeometry);
   
   nominal_gain_ = lutMetadata_->getNominalGain();
   float rctlsb =lutMetadata_->getRctLsb();
   if (rctlsb != 0.25 && rctlsb != 0.5)
      throw cms::Exception("RCTLSB") << " value=" << rctlsb << " (should be 0.25 or 0.5)" << std::endl;
   rctlsb_factor_ = rctlsb;

   if (compressionFile_.empty() && decompressionFile_.empty()) {
      loadHCALCompress();
   }
   else {
      // TODO Message to discourage using txt.
      std::cout << "From Text File:" << std::endl;
      loadHCALCompress(compressionFile_);
   }
   isLoaded_ = true;
}

void CaloTPGTranscoderULUT::printDecompression() const{
   std::cout << "RCT Decompression table" << std::endl;                          
   for (int i=0; i < 256; i++) {                                                 
      for (int j=1; j <= theTrigTowerGeometry->nTowers(); j++)
         std::cout << int(hcaletValue(j,i)*100. + 0.5)/100. << " ";
      std::cout << std::endl;                                               
      for (int j=1; j <= theTrigTowerGeometry->nTowers(); j++)               
         if (hcaletValue(j,i) != hcaletValue(-j,i))                    
            cout << "Error: decompression table for ieta = +/- " << j << " disagree! " << hcaletValue(-j,i) << ", " << hcaletValue(j,i) << endl;        
   }
}

//int CaloTPGTranscoderULUT::getLutGranularity(const DetId& id) const{
//   int ieta = id.ietaAbs();
//   if (ieta < 18) return 1;
//   else if (ieta < 27) return 2;
//   else if (ieta < 29) return 5;
//   return 0;
//}
//
//int CaloTPGTranscoderULUT::getLutThreshold(const DetId& id) const{
//   int ieta = id.ietaAbs();
//   if (ieta < 18) return 4;
//   else if (ieta < 27) return 2;
//   else if (ieta < 29) return 1;
//   return 0;
//}
//

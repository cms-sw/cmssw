#include "CalibCalorimetry/CaloTPG/src/CaloTPGTranscoderULUT.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
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
                                                : nominal_gain_(0.), rctlsb_factor_(0.),
                                                  compressionFile_(compressionFile),
                                                  decompressionFile_(decompressionFile)
{
  outputLUT_.clear();
}

CaloTPGTranscoderULUT::~CaloTPGTranscoderULUT() {
}

void CaloTPGTranscoderULUT::loadHCALCompress(HcalLutMetadata const& lutMetadata,
                                             HcalTrigTowerGeometry const& theTrigTowerGeometry) {
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
 
    std::vector<DetId> allChannels = lutMetadata.getAllChannels();

    for(std::vector<DetId>::iterator i=allChannels.begin(); i!=allChannels.end(); ++i){

	if(HcalDetId(*i).subdet()!=HcalTriggerTower) continue;
	
	HcalTrigTowerDetId id(*i); 
	if(!theTopology->validHT(HcalTrigTowerDetId(id))) continue;


	unsigned int index = getOutputLUTId(id); 

	if(index >= outputLUT_.size()){
	    outputLUT_.resize(index+1);
	    hcaluncomp_.resize(index+1);
	}

	const HcalLutMetadatum *meta = lutMetadata.getValues(id);
	unsigned int threshold	     = meta->getOutputLutThreshold();

	int ieta=id.ieta();
	bool isHBHE = (abs(ieta) < theTrigTowerGeometry.firstHFTower()); 

	for (unsigned int i = 0; i < threshold; ++i) outputLUT_[index].push_back(0);
	for (unsigned int i = threshold; i < OUTPUT_LUT_SIZE; ++i){
	    LUT value =  isHBHE ? analyticalLUT[i] : identityLUT[i];
	    outputLUT_[index].push_back(value);
        }

	//now uncompression LUTs
	hcaluncomp_[index].resize(TPGMAX);

	double eta_low = 0., eta_high = 0.;
	theTrigTowerGeometry.towerEtaBounds(ieta,eta_low,eta_high); 
	double cosh_ieta   = fabs(cosh((eta_low + eta_high)/2.));
	double granularity =  meta->getLutGranularity(); 

	double factor = isHBHE ?  (nominal_gain_ / cosh_ieta * granularity) : rctlsb_factor_;

        LUT tpg = outputLUT_[index][0];
        int low = 0;
        for (unsigned int i = 0; i < OUTPUT_LUT_SIZE; ++i){
	    if (outputLUT_[index][i] != tpg){
               unsigned int mid = (low + i)/2; 
               hcaluncomp_[index][tpg] = (tpg == 0 ? low : factor * mid);
               low = i;
               tpg = outputLUT_[index][i];
            }
        }
        hcaluncomp_[index][tpg] = factor * low;
    }
}

HcalTriggerPrimitiveSample CaloTPGTranscoderULUT::hcalCompress(const HcalTrigTowerDetId& id, unsigned int sample, bool fineGrain) const {
  int itower = getOutputLUTId(id);

  if (sample >= OUTPUT_LUT_SIZE) {
    throw cms::Exception("Out of Range") << "LUT has 1024 entries for " << itower << " but " << sample << " was requested.";
    sample=OUTPUT_LUT_SIZE - 1;
  }

  return HcalTriggerPrimitiveSample(outputLUT_[itower][sample],fineGrain,0,0);
}

double CaloTPGTranscoderULUT::hcaletValue(const int& ieta, const int& iphi, const int& compET) const {
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
  int compET = hc.compressedEt();	// to be within the range by the class
  int itower = getOutputLUTId(hid);
  double etvalue = hcaluncomp_[itower][compET];
  return etvalue;
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

int CaloTPGTranscoderULUT::getOutputLUTId(const HcalTrigTowerDetId& id) const {
    return theTopology->detId2denseIdHT(id);
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

std::vector<unsigned int> CaloTPGTranscoderULUT::getCompressionLUT(HcalTrigTowerDetId id) const {
   int itower = getOutputLUTId(id);
   return outputLUT_[itower];
}

void CaloTPGTranscoderULUT::setup(HcalLutMetadata const& lutMetadata, HcalTrigTowerGeometry const& theTrigTowerGeometry)
{
    theTopology = lutMetadata.topo();
    nominal_gain_ = lutMetadata.getNominalGain();
    float rctlsb =lutMetadata.getRctLsb();
    if (rctlsb != 0.25 && rctlsb != 0.5)
	throw cms::Exception("RCTLSB") << " value=" << rctlsb << " (should be 0.25 or 0.5)" << std::endl;
    rctlsb_factor_ = rctlsb;

    if (compressionFile_.empty() && decompressionFile_.empty()) {
	loadHCALCompress(lutMetadata,theTrigTowerGeometry);
    }
    else {
	throw cms::Exception("Not Implemented") << "setup of CaloTPGTranscoderULUT from text files";
   }
}


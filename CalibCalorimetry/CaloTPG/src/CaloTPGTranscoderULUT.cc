#include "CalibCalorimetry/CaloTPG/src/CaloTPGTranscoderULUT.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/HcalTowerAlgo/interface/HcalTrigTowerGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include <iostream>
#include <fstream>
#include <math.h>

//#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CondFormats/DataRecord/interface/HcalLutMetadataRcd.h"
#include "CalibCalorimetry/HcalTPGAlgos/interface/HcaluLUTTPGCoder.h"

using namespace std;

CaloTPGTranscoderULUT::CaloTPGTranscoderULUT(const std::string& compressionFile,
                                             const std::string& decompressionFile)
                                                : theTopology(0),
                                                  nominal_gain_(0.), lsb_factor_(0.), rct_factor_(1.), nct_factor_(1.),
                                                  compressionFile_(compressionFile),
                                                  decompressionFile_(decompressionFile),
						  size(0)
{
  outputLUT_.clear();
}

CaloTPGTranscoderULUT::~CaloTPGTranscoderULUT() {
}

void CaloTPGTranscoderULUT::loadHCALCompress(HcalLutMetadata const& lutMetadata,
                                             HcalTrigTowerGeometry const& theTrigTowerGeometry) {
    // Initialize analytical compression LUT's here
    if (OUTPUT_LUT_SIZE != (unsigned int) 0x400)
        edm::LogError("CaloTPGTranscoderULUT") << "Analytic compression expects 10-bit LUT; found LUT with " << OUTPUT_LUT_SIZE << " entries instead";

    if (!theTopology) {
        throw cms::Exception("CaloTPGTranscoderULUT") << "Topology not set! Use CaloTPGTranscoderULUT::setup(...) first!";
    }

    std::array<unsigned int, OUTPUT_LUT_SIZE> analyticalLUT;
    std::array<unsigned int, OUTPUT_LUT_SIZE> linearRctLUT;
    std::array<unsigned int, OUTPUT_LUT_SIZE> linearNctLUT;

    // Compute compression LUT
    for (unsigned int i=0; i < OUTPUT_LUT_SIZE; i++) {
	analyticalLUT[i] = (unsigned int)(sqrt(14.94*log(1.+i/14.94)*i) + 0.5);
	linearRctLUT[i] = min((unsigned int)(i/rct_factor_), TPGMAX - 1);
	linearNctLUT[i] = min((unsigned int)(i/nct_factor_), TPGMAX - 1);
    }
 
    std::vector<DetId> allChannels = lutMetadata.getAllChannels();

    for(std::vector<DetId>::iterator i=allChannels.begin(); i!=allChannels.end(); ++i){

	if (not HcalGenericDetId(*i).isHcalTrigTowerDetId()) {
	    if (not HcalGenericDetId(*i).isHcalDetId())
		edm::LogWarning("CaloTPGTranscoderULUT") << "Encountered invalid HcalDetId " << HcalGenericDetId(*i);
	    continue;
	}
	
	HcalTrigTowerDetId id(*i); 
	if(!theTopology->validHT(id)) continue;


	unsigned int index = getOutputLUTId(id); 

	if(index >= size){
	    size=index+1;
	    outputLUT_.resize(size);
	    hcaluncomp_.resize(size);
	}

	const HcalLutMetadatum *meta = lutMetadata.getValues(id);
	unsigned int threshold	     = meta->getOutputLutThreshold();

	int ieta=id.ieta();
	int version=id.version();
	bool isHBHE = (abs(ieta) < theTrigTowerGeometry.firstHFTower(version)); 

	for (unsigned int i = 0; i < threshold; ++i) outputLUT_[index].push_back(0);
	for (unsigned int i = threshold; i < OUTPUT_LUT_SIZE; ++i){
	    LUT value =  isHBHE ? analyticalLUT[i] : 
			 (version==0?linearRctLUT[i]:linearNctLUT[i]);
	    outputLUT_[index].push_back(value);
        }

	//now uncompression LUTs
	hcaluncomp_[index].resize(TPGMAX);

	double eta_low = 0., eta_high = 0.;
	theTrigTowerGeometry.towerEtaBounds(ieta,version,eta_low,eta_high); 
	double cosh_ieta   = fabs(cosh((eta_low + eta_high)/2.));
	double granularity =  meta->getLutGranularity(); 

	if(isHBHE){
	    double factor = nominal_gain_ / cosh_ieta * granularity;
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
	else{
	    LUT tpg = outputLUT_[index][0];
	    hcaluncomp_[index][tpg]=0;
	    for (unsigned int i = 0; i < OUTPUT_LUT_SIZE; ++i){
		if (outputLUT_[index][i] != tpg){
		   tpg = outputLUT_[index][i];
		   hcaluncomp_[index][tpg] = lsb_factor_ * i / (version==0?rct_factor_:nct_factor_);
		}
	    }
	}
    }
}

HcalTriggerPrimitiveSample CaloTPGTranscoderULUT::hcalCompress(const HcalTrigTowerDetId& id, unsigned int sample, bool fineGrain) const {
  unsigned int itower = getOutputLUTId(id);

  if (sample >= OUTPUT_LUT_SIZE) {
    throw cms::Exception("Out of Range") << "LUT has 1024 entries for " << itower << " but " << sample << " was requested.";
    sample=OUTPUT_LUT_SIZE - 1;
  }

  if(itower >= size){
    throw cms::Exception("Out of Range") << "No decompression LUT found for " << id;
  }

  return HcalTriggerPrimitiveSample(outputLUT_[itower][sample],fineGrain,0,0);
}

double CaloTPGTranscoderULUT::hcaletValue(const int& ieta, const int& iphi, const int& compET) const {
  double etvalue = 0.;
  int itower = getOutputLUTId(ieta,iphi);
  if (itower < 0) {
    edm::LogError("CaloTPGTranscoderULUT") << "No decompression LUT found for ieta, iphi = " << ieta << ", " << iphi;
  } else if (compET < 0 || compET >= (int) TPGMAX) {
    edm::LogError("CaloTPGTranscoderULUT") << "Compressed value out of range: eta, phi, cET = " << ieta << ", " << iphi << ", " << compET;
  } else {
    etvalue = hcaluncomp_[itower][compET];
  }
  return(etvalue);
}

double CaloTPGTranscoderULUT::hcaletValue(const int& ieta, const int& compET) const {
// This is now an obsolete method; we return the AVERAGE over all the allowed iphi channels if it's invoked
// The user is encouraged to use hcaletValue(const int& ieta, const int& iphi, const int& compET) instead

  double etvalue = 0.;
  if (compET < 0 || compET >= (int) TPGMAX) {
    edm::LogError("CaloTPGTranscoderULUT") << "Compressed value out of range: eta, cET = " << ieta << ", " << compET;
  } else {
	int nphi = 0;
	for (int iphi=1; iphi <= 72; iphi++) {
		if (HTvalid(ieta,iphi)) {
			nphi++;
			int itower = getOutputLUTId(ieta,iphi);
			etvalue += hcaluncomp_[itower][compET];
		}
	}
	if (nphi > 0) {
		etvalue /= nphi;
	} else {
		edm::LogError("CaloTPGTranscoderULUT") << "No decompression LUTs found for any iphi for ieta = " << ieta;
	}
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
	HcalTrigTowerDetId id(ieta, iphiin);
	if (!theTopology) {
		throw cms::Exception("CaloTPGTranscoderULUT") << "Topology not set! Use CaloTPGTranscoderULUT::setup(...) first!";
	}
	return theTopology->validHT(id);
}

int CaloTPGTranscoderULUT::getOutputLUTId(const HcalTrigTowerDetId& id) const {
    if (!theTopology) {
        throw cms::Exception("CaloTPGTranscoderULUT") << "Topology not set! Use CaloTPGTranscoderULUT::setup(...) first!";
    }
    return theTopology->detId2denseIdHT(id);
}

int CaloTPGTranscoderULUT::getOutputLUTId(const int ieta, const int iphiin) const {
	if (!theTopology) {
		throw cms::Exception("CaloTPGTranscoderULUT") << "Topology not set! Use CaloTPGTranscoderULUT::setup(...) first!";
	}
	HcalTrigTowerDetId id(ieta, iphiin);
	return theTopology->detId2denseIdHT(id);
}

const std::vector<unsigned int>& CaloTPGTranscoderULUT::getCompressionLUT(const HcalTrigTowerDetId& id) const {
   int itower = getOutputLUTId(id);
   return outputLUT_[itower];
}

void CaloTPGTranscoderULUT::setup(HcalLutMetadata const& lutMetadata, HcalTrigTowerGeometry const& theTrigTowerGeometry, int nctScaleShift, int rctScaleShift)
{
    theTopology	    = lutMetadata.topo();
    nominal_gain_   = lutMetadata.getNominalGain();
    lsb_factor_	    = lutMetadata.getRctLsb();

    rct_factor_  = lsb_factor_/(HcaluLUTTPGCoder::lsb_*(1<<rctScaleShift));
    nct_factor_  = lsb_factor_/(HcaluLUTTPGCoder::lsb_*(1<<nctScaleShift));

    if (compressionFile_.empty() && decompressionFile_.empty()) {
	loadHCALCompress(lutMetadata,theTrigTowerGeometry);
    }
    else {
	throw cms::Exception("Not Implemented") << "setup of CaloTPGTranscoderULUT from text files";
   }
}

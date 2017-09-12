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
                                                  decompressionFile_(decompressionFile)
{
}

CaloTPGTranscoderULUT::~CaloTPGTranscoderULUT() {
}

void CaloTPGTranscoderULUT::loadHCALCompress(HcalLutMetadata const& lutMetadata,
                                             HcalTrigTowerGeometry const& theTrigTowerGeometry) {
    if (!theTopology) {
        throw cms::Exception("CaloTPGTranscoderULUT") << "Topology not set! Use CaloTPGTranscoderULUT::setup(...) first!";
    }

    std::array<unsigned int, OUTPUT_LUT_SIZE> analyticalLUT;
    std::array<unsigned int, OUTPUT_LUT_SIZE> linearRctLUT;
    std::array<unsigned int, OUTPUT_LUT_SIZE> linearNctLUT;

    // Compute compression LUT
    for (unsigned int i=0; i < OUTPUT_LUT_SIZE; i++) {
	analyticalLUT[i] = min((unsigned int)(sqrt(14.94*log(1.+i/14.94)*i) + 0.5), TPGMAX - 1);
	linearRctLUT[i] = min((unsigned int)(i/rct_factor_), TPGMAX - 1);
	linearNctLUT[i] = min((unsigned int)(i/nct_factor_), TPGMAX - 1);
    }
 
    std::vector<DetId> allChannels = lutMetadata.getAllChannels();

    for(std::vector<DetId>::iterator i=allChannels.begin(); i!=allChannels.end(); ++i){

	if (not HcalGenericDetId(*i).isHcalTrigTowerDetId()) {
	  if ((not HcalGenericDetId(*i).isHcalDetId()) and
	      (not HcalGenericDetId(*i).isHcalZDCDetId()) and
	      (not HcalGenericDetId(*i).isHcalCastorDetId()))
	    edm::LogWarning("CaloTPGTranscoderULUT") << "Encountered invalid HcalDetId " << HcalGenericDetId(*i);
	  continue; 
	}
	
	HcalTrigTowerDetId id(*i); 
	if(!theTopology->validHT(id)) continue;

	unsigned int index = getOutputLUTId(id); 

	const HcalLutMetadatum *meta = lutMetadata.getValues(id);
	unsigned int threshold	     = meta->getOutputLutThreshold();

	int ieta=id.ieta();
	int version=id.version();
	bool isHBHE = (abs(ieta) < theTrigTowerGeometry.firstHFTower(version)); 

        unsigned int lutsize = getOutputLUTSize(id);
	outputLUT_[index].resize(lutsize);

        for (unsigned int i = 0; i < threshold; ++i)
           outputLUT_[index][i] = 0;

        if (isHBHE) {
           for (unsigned int i = threshold; i < lutsize; ++i)
              outputLUT_[index][i] = analyticalLUT[i];
	} else {
           for (unsigned int i = threshold; i < lutsize; ++i)
              outputLUT_[index][i] = version == 0 ? linearRctLUT[i] : linearNctLUT[i];
        }

	double eta_low = 0., eta_high = 0.;
	theTrigTowerGeometry.towerEtaBounds(ieta,version,eta_low,eta_high); 
	double cosh_ieta   = fabs(cosh((eta_low + eta_high)/2.));
	double granularity =  meta->getLutGranularity(); 

	if(isHBHE){
	    double factor = nominal_gain_ / cosh_ieta * granularity;
	    LUT tpg = outputLUT_[index][0];
	    int low = 0;
	    for (unsigned int i = 0; i < getOutputLUTSize(id); ++i){
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
	    for (unsigned int i = 0; i < getOutputLUTSize(id); ++i){
		if (outputLUT_[index][i] != tpg){
		   tpg = outputLUT_[index][i];
		   hcaluncomp_[index][tpg] = lsb_factor_ * i / (version==0?rct_factor_:nct_factor_);
		}
	    }
	}
    }

}

HcalTriggerPrimitiveSample CaloTPGTranscoderULUT::hcalCompress(const HcalTrigTowerDetId& id, unsigned int sample, int fineGrain) const {
  unsigned int itower = getOutputLUTId(id);

  if (sample >= getOutputLUTSize(id))
    throw cms::Exception("Out of Range")
       << "LUT has " << getOutputLUTSize(id) << " entries for " << id << " but " << sample << " was requested.";

  if(itower >= outputLUT_.size())
    throw cms::Exception("Out of Range") << "No decompression LUT found for " << id;

  return HcalTriggerPrimitiveSample(outputLUT_[itower][sample], fineGrain);
}

double CaloTPGTranscoderULUT::hcaletValue(const int& ieta, const int& iphi, const int& version, const int& compET) const {
  double etvalue = 0.;
  int itower = getOutputLUTId(ieta,iphi, version);
  if (itower < 0) {
    edm::LogError("CaloTPGTranscoderULUT") << "No decompression LUT found for ieta, iphi = " << ieta << ", " << iphi;
  } else if (compET < 0 || compET >= (int) TPGMAX) {
    edm::LogError("CaloTPGTranscoderULUT") << "Compressed value out of range: eta, phi, cET = " << ieta << ", " << iphi << ", " << compET;
  } else {
    etvalue = hcaluncomp_[itower][compET];
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

bool CaloTPGTranscoderULUT::HTvalid(const int ieta, const int iphiin, const int version) const {
	HcalTrigTowerDetId id(ieta, iphiin);
	id.setVersion(version);
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

int CaloTPGTranscoderULUT::getOutputLUTId(const int ieta, const int iphiin, const int version) const {
	if (!theTopology) {
		throw cms::Exception("CaloTPGTranscoderULUT") << "Topology not set! Use CaloTPGTranscoderULUT::setup(...) first!";
	}
	HcalTrigTowerDetId id(ieta, iphiin);
	id.setVersion(version);
	return theTopology->detId2denseIdHT(id);
}

unsigned int
CaloTPGTranscoderULUT::getOutputLUTSize(const HcalTrigTowerDetId& id) const
{
   if (!theTopology)
      throw cms::Exception("CaloTPGTranscoderULUT")
         << "Topology not set! Use CaloTPGTranscoderULUT::setup(...) first!";

   switch (theTopology->triggerMode()) {
      case HcalTopologyMode::TriggerMode_2009:
      case HcalTopologyMode::TriggerMode_2016:
         return QIE8_OUTPUT_LUT_SIZE;
      case HcalTopologyMode::TriggerMode_2017:
         if (id.ietaAbs() <= theTopology->lastHERing())
            return QIE8_OUTPUT_LUT_SIZE;
         else
            return QIE10_OUTPUT_LUT_SIZE;
      case HcalTopologyMode::TriggerMode_2017plan1:
         if (plan1_towers_.find(id) != plan1_towers_.end())
            return QIE11_OUTPUT_LUT_SIZE;
         else if (id.ietaAbs() <= theTopology->lastHERing())
            return QIE8_OUTPUT_LUT_SIZE;
         else
            return QIE10_OUTPUT_LUT_SIZE;
      case HcalTopologyMode::TriggerMode_2018legacy:
      case HcalTopologyMode::TriggerMode_2018:
         if (id.ietaAbs() <= theTopology->lastHBRing())
            return QIE8_OUTPUT_LUT_SIZE;
         else if (id.ietaAbs() <= theTopology->lastHERing())
            return QIE11_OUTPUT_LUT_SIZE;
         else
            return QIE10_OUTPUT_LUT_SIZE;
      case HcalTopologyMode::TriggerMode_2019:
         if (id.ietaAbs() <= theTopology->lastHERing())
            return QIE11_OUTPUT_LUT_SIZE;
         else
            return QIE10_OUTPUT_LUT_SIZE;
      default:
         throw cms::Exception("CaloTPGTranscoderULUT")
            << "Unknown trigger mode used by the topology!";
   }
}

const std::vector<unsigned int> CaloTPGTranscoderULUT::getCompressionLUT(const HcalTrigTowerDetId& id) const {
   int itower = getOutputLUTId(id);
   auto lut = outputLUT_[itower];
   std::vector<unsigned int> result(lut.begin(), lut.end());
   return result;
}

void CaloTPGTranscoderULUT::setup(HcalLutMetadata const& lutMetadata, HcalTrigTowerGeometry const& theTrigTowerGeometry, int nctScaleShift, int rctScaleShift)
{
    theTopology	    = lutMetadata.topo();
    nominal_gain_   = lutMetadata.getNominalGain();
    lsb_factor_	    = lutMetadata.getRctLsb();

    rct_factor_  = lsb_factor_/(HcaluLUTTPGCoder::lsb_*(1<<rctScaleShift));
    nct_factor_  = lsb_factor_/(HcaluLUTTPGCoder::lsb_*(1<<nctScaleShift));

    outputLUT_.resize(theTopology->getHTSize());
    hcaluncomp_.resize(theTopology->getHTSize());

    plan1_towers_.clear();
    for (const auto& id: lutMetadata.getAllChannels()) {
       if (not (id.det() == DetId::Hcal and theTopology->valid(id)))
          continue;
       HcalDetId cell(id);
       if (not theTopology->dddConstants()->isPlan1(cell))
          continue;
       for (const auto& tower: theTrigTowerGeometry.towerIds(cell))
          plan1_towers_.emplace(tower);
    }

    if (compressionFile_.empty() && decompressionFile_.empty()) {
	loadHCALCompress(lutMetadata,theTrigTowerGeometry);
    }
    else {
	throw cms::Exception("Not Implemented") << "setup of CaloTPGTranscoderULUT from text files";
    }
}

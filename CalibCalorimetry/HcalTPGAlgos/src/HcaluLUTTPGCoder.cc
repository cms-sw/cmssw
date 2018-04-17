#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include "CalibCalorimetry/HcalTPGAlgos/interface/HcaluLUTTPGCoder.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "DataFormats/HcalDigi/interface/HcalQIENum.h"
#include "DataFormats/HcalDigi/interface/QIE10DataFrame.h"
#include "DataFormats/HcalDigi/interface/QIE11DataFrame.h"
#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include "Geometry/HcalTowerAlgo/interface/HcalTrigTowerGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CalibCalorimetry/CaloTPG/src/CaloTPGTranscoderULUT.h"
#include "CondFormats/HcalObjects/interface/HcalL1TriggerObjects.h"
#include "CondFormats/HcalObjects/interface/HcalL1TriggerObject.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalSiPMnonlinearity.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseContainmentCorrection.h"
#include "CalibCalorimetry/HcalTPGAlgos/interface/XMLProcessor.h"
#include "CalibCalorimetry/HcalTPGAlgos/interface/LutXml.h"

const float HcaluLUTTPGCoder::lsb_=1./16;

const int HcaluLUTTPGCoder::QIE8_LUT_BITMASK;
const int HcaluLUTTPGCoder::QIE10_LUT_BITMASK;
const int HcaluLUTTPGCoder::QIE11_LUT_BITMASK;

constexpr double MaximumFractionalError = 0.002; // 0.2% error allowed from this source

HcaluLUTTPGCoder::HcaluLUTTPGCoder(const HcalTopology* top, const edm::ESHandle<HcalTimeSlew>& delay) : topo_(top),  delay_(delay), LUTGenerationMode_(true), bitToMask_(0), allLinear_(false), linearLSB_QIE8_(1.), linearLSB_QIE11_(1.), pulseCorr_(std::make_unique<HcalPulseContainmentManager>(MaximumFractionalError)) {
  firstHBEta_ = topo_->firstHBRing();      
  lastHBEta_  = topo_->lastHBRing();
  nHBEta_     = (lastHBEta_-firstHBEta_+1);
  maxDepthHB_ = topo_->maxDepth(HcalBarrel);
  sizeHB_     = 2*nHBEta_*nFi_*maxDepthHB_;
  firstHEEta_ = topo_->firstHERing();
  lastHEEta_  = topo_->lastHERing();
  nHEEta_     = (lastHEEta_-firstHEEta_+1);
  maxDepthHE_ = topo_->maxDepth(HcalEndcap);
  sizeHE_     = 2*nHEEta_*nFi_*maxDepthHE_;
  firstHFEta_ = topo_->firstHFRing();
  lastHFEta_  = topo_->lastHFRing();
  nHFEta_     = (lastHFEta_-firstHFEta_+1);
  maxDepthHF_ = topo_->maxDepth(HcalForward);
  sizeHF_     = 2*nHFEta_*nFi_*maxDepthHF_;
  size_t nluts= (size_t)(sizeHB_+sizeHE_+sizeHF_+1);
  inputLUT_   = std::vector<HcaluLUTTPGCoder::Lut>(nluts);
  gain_       = std::vector<float>(nluts, 0.);
  ped_        = std::vector<float>(nluts, 0.);
  make_cosh_ieta_map();
}

void HcaluLUTTPGCoder::compress(const IntegerCaloSamples& ics, const std::vector<bool>& featureBits, HcalTriggerPrimitiveDigi& tp) const {
  throw cms::Exception("PROBLEM: This method should never be invoked!");
}

HcaluLUTTPGCoder::~HcaluLUTTPGCoder() {
}

int HcaluLUTTPGCoder::getLUTId(HcalSubdetector id, int ieta, int iphi, int depth) const {
  int retval(0);
  if (id == HcalBarrel) {
    retval = (depth-1)+maxDepthHB_*(iphi-1);
    if (ieta>0) retval+=maxDepthHB_*nFi_*(ieta-firstHBEta_);
    else        retval+=maxDepthHB_*nFi_*(ieta+lastHBEta_+nHBEta_);
  } else if (id == HcalEndcap) {
    retval = sizeHB_;
    retval+= (depth-1)+maxDepthHE_*(iphi-1);
    if (ieta>0) retval+=maxDepthHE_*nFi_*(ieta-firstHEEta_);
    else        retval+=maxDepthHE_*nFi_*(ieta+lastHEEta_+nHEEta_);
  } else if (id == HcalForward) {
    retval = sizeHB_+sizeHE_;
    retval+= (depth-1)+maxDepthHF_*(iphi-1);
    if (ieta>0) retval+=maxDepthHF_*nFi_*(ieta-firstHFEta_);
    else        retval+=maxDepthHF_*nFi_*(ieta+lastHFEta_+nHFEta_);
  }
  return retval;
}

int HcaluLUTTPGCoder::getLUTId(uint32_t rawid) const {
  HcalDetId detid(rawid);
  return getLUTId(detid.subdet(), detid.ieta(), detid.iphi(), detid.depth());
}

int HcaluLUTTPGCoder::getLUTId(const HcalDetId& detid) const {
  return getLUTId(detid.subdet(), detid.ieta(), detid.iphi(), detid.depth());
}

void HcaluLUTTPGCoder::update(const char* filename, bool appendMSB) {

  std::ifstream file(filename, std::ios::in);
  assert(file.is_open());

  std::vector<HcalSubdetector> subdet;
  std::string buffer;

  // Drop first (comment) line
  std::getline(file, buffer);
  std::getline(file, buffer);

  unsigned int index = buffer.find("H", 0);
  while (index < buffer.length()){
    std::string subdetStr = buffer.substr(index, 2);
    if (subdetStr == "HB") subdet.push_back(HcalBarrel);
    else if (subdetStr == "HE") subdet.push_back(HcalEndcap);
    else if (subdetStr == "HF") subdet.push_back(HcalForward);
    //TODO Check subdet
    //else exception
    index += 2;
    index = buffer.find("H", index);
  }

  // Get upper/lower ranges for ieta/iphi/depth
  size_t nCol = subdet.size();
  assert(nCol > 0);

  std::vector<int> ietaU;
  std::vector<int> ietaL;
  std::vector<int> iphiU;
  std::vector<int> iphiL;
  std::vector<int> depU;
  std::vector<int> depL;
  std::vector< Lut > lutFromFile(nCol);
  LutElement lutValue;

  for (size_t i=0; i<nCol; ++i) {
    int ieta;
    file >> ieta;
    ietaL.push_back(ieta);
  }

  for (size_t i=0; i<nCol; ++i) {
    int ieta;
    file >> ieta;
    ietaU.push_back(ieta);
  }

  for (size_t i=0; i<nCol; ++i) {
    int iphi;
    file >> iphi;
    iphiL.push_back(iphi);
  }

  for (size_t i=0; i<nCol; ++i) {
    int iphi;
    file >> iphi;
    iphiU.push_back(iphi);
  }

  for (size_t i=0; i<nCol; ++i) {
    int dep;
    file >> dep;
    depL.push_back(dep);
  }

  for (size_t i=0; i<nCol; ++i) {
    int dep;
    file >> dep;
    depU.push_back(dep);
  }
  
  // Read Lut Entry
  for (size_t i=0; file >> lutValue; i = (i+1) % nCol){
    lutFromFile[i].push_back(lutValue);
  }

  // Check lut size
  for (size_t i=0; i<nCol; ++i) assert(lutFromFile[i].size() == INPUT_LUT_SIZE);

  for (size_t i=0; i<nCol; ++i){
    for (int ieta = ietaL[i]; ieta <= ietaU[i]; ++ieta){
      for (int iphi = iphiL[i]; iphi <= iphiU[i]; ++iphi){
	for (int depth = depL[i]; depth <= depU[i]; ++depth){
	      
	  HcalDetId id(subdet[i], ieta, iphi, depth);
	  if (!topo_->valid(id)) continue;

	  int lutId = getLUTId(id);
	  for (size_t adc = 0; adc < INPUT_LUT_SIZE; ++adc){
	    if (appendMSB){
	      // Append FG bit LUT to MSB
	      // MSB = Most Significant Bit = bit 10
	      // Overwrite bit 10
	      LutElement msb = (lutFromFile[i][adc] != 0 ? QIE8_LUT_MSB : 0);
	      inputLUT_[lutId][adc] = (msb | (inputLUT_[lutId][adc] & QIE8_LUT_BITMASK));
	    }
	    else inputLUT_[lutId][adc] = lutFromFile[i][adc];
	  }// for adc
	}// for depth
      }// for iphi
    }// for ieta
  }// for nCol
}

void HcaluLUTTPGCoder::updateXML(const char* filename) {
  LutXml * _xml = new LutXml(filename);
  _xml->create_lut_map();
  HcalSubdetector subdet[3] = {HcalBarrel, HcalEndcap, HcalForward};
  for (int ieta = -HcalDetId::kHcalEtaMask2; 
       ieta <= HcalDetId::kHcalEtaMask2; ++ieta) {
    for (int iphi = 0; iphi <= HcalDetId::kHcalPhiMask2; ++iphi) {
      for (int depth = 1; depth < HcalDetId::kHcalDepthMask2; ++depth) {
	for (int isub=0; isub<3; ++isub) {
	  HcalDetId detid(subdet[isub], ieta, iphi, depth);
	  if (!topo_->valid(detid)) continue;
	  int id = getLUTId(subdet[isub], ieta, iphi, depth);
	  std::vector<unsigned int>* lut = _xml->getLutFast(detid);
	  if (lut==nullptr) throw cms::Exception("PROBLEM: No inputLUT_ in xml file for ") << detid << std::endl;
	  if (lut->size()!=INPUT_LUT_SIZE) throw cms::Exception ("PROBLEM: Wrong inputLUT_ size in xml file for ") << detid << std::endl;
	  for (unsigned int i=0; i<INPUT_LUT_SIZE; ++i) inputLUT_[id][i] = (LutElement)lut->at(i);
	}
      }
    }
  }
  delete _xml;
  XMLProcessor::getInstance()->terminate();
}

double HcaluLUTTPGCoder::cosh_ieta(int ieta, int depth, HcalSubdetector subdet) {
  // ieta = 28 and 29 are both associated with trigger tower 28
  // so special handling is required. HF ieta=29 channels included in TT30
  // are already handled correctly in cosh_ieta_
  if (abs(ieta) >= 28 && subdet == HcalEndcap && allLinear_) {
    if (abs(ieta) == 29) return cosh_ieta_29_HE_;
    if (abs(ieta) == 28) {
      if (depth <= 3) return cosh_ieta_28_HE_low_depths_;
      else return cosh_ieta_28_HE_high_depths_;
    }
  }

  return cosh_ieta_[ieta];
}

void HcaluLUTTPGCoder::make_cosh_ieta_map(void) {

  cosh_ieta_ = std::vector<double>(lastHFEta_ + 1, -1.0);

  HcalTrigTowerGeometry triggeo(topo_);

  for (int i = 1; i <= firstHFEta_; ++i) {
    double eta_low = 0., eta_high = 0.;
    triggeo.towerEtaBounds(i, 0, eta_low, eta_high);
    cosh_ieta_[i] = cosh((eta_low + eta_high)/2.);
  }
  for (int i = firstHFEta_; i <= lastHFEta_; ++i){
    std::pair<double,double> etas = topo_->etaRange(HcalForward,i);
    double eta1 = etas.first;
    double eta2 = etas.second;
    cosh_ieta_[i] = cosh((eta1 + eta2)/2.);
  }

  // trigger tower 28 in HE has a more complicated geometry
  std::pair<double, double> eta28 = topo_->etaRange(HcalEndcap, 28);
  std::pair<double, double> eta29 = topo_->etaRange(HcalEndcap, 29);
  cosh_ieta_29_HE_ = cosh((eta29.first + eta29.second)/2.);
  cosh_ieta_28_HE_low_depths_ = cosh((eta28.first + eta28.second)/2.);
  // for higher depths in ieta = 28, the trigger tower extends past
  // the ieta = 29 channels
  cosh_ieta_28_HE_high_depths_ = cosh((eta28.first + eta29.second)/2.);
}

void HcaluLUTTPGCoder::update(const HcalDbService& conditions) {
    
    HcalCalibrations calibrations;
    const HcalLutMetadata *metadata = conditions.getHcalLutMetadata();
    assert(metadata !=nullptr);
    float nominalgain_ = metadata->getNominalGain();

    pulseCorr_->beginRun(&conditions, delay_);

    make_cosh_ieta_map();

    for (const auto& id: metadata->getAllChannels()) {
     
	if (not (id.det() == DetId::Hcal and topo_->valid(id))) continue;

	HcalDetId cell(id);
	HcalSubdetector subdet = cell.subdet();
     
	if (subdet != HcalBarrel and subdet != HcalEndcap and subdet != HcalForward) continue;

	const HcalQIECoder* channelCoder = conditions.getHcalCoder (cell);
	const HcalQIEShape* shape = conditions.getHcalShape(cell);
	HcalCoderDb coder (*channelCoder, *shape);
	const HcalLutMetadatum *meta = metadata->getValues(cell);

	unsigned int mipMax = 0;
	unsigned int mipMin = 0;
     
	bool is2018OrLater = topo_->triggerMode() >= HcalTopologyMode::TriggerMode_2018 or
	  topo_->triggerMode() == HcalTopologyMode::TriggerMode_2018legacy;
	if (is2018OrLater or topo_->dddConstants()->isPlan1(cell)) {
	    const HcalTPChannelParameter *channelParameters = conditions.getHcalTPChannelParameter(cell);
	    mipMax = channelParameters->getFGBitInfo() >> 16;
	    mipMin = channelParameters->getFGBitInfo() & 0xFFFF;
	}

	int lutId = getLUTId(cell);
	Lut &lut=inputLUT_[lutId]; 
	float ped = 0;
	float gain = 0;
	uint32_t status = 0;

	if (LUTGenerationMode_){
	    const HcalCalibrations& calibrations = conditions.getHcalCalibrations(cell);
	    for (auto capId : {0,1,2,3}){
		ped += calibrations.effpedestal(capId);
		gain += calibrations.LUTrespcorrgain(capId);
	    }
	    ped /= 4.0;
	    gain /= 4.0;

	    //Get Channel Quality
	    const HcalChannelStatus* channelStatus = conditions.getHcalChannelStatus(cell);
	    status = channelStatus->getValue();
     
	} else {
	    const HcalL1TriggerObject* myL1TObj = conditions.getHcalL1TriggerObject(cell);
	    ped = myL1TObj->getPedestal();
	    gain = myL1TObj->getRespGain();
	    status = myL1TObj->getFlag();
	} // LUTGenerationMode_

	ped_[lutId] = ped;
	gain_[lutId] = gain;
	bool isMasked = ( (status & bitToMask_) > 0 );
	float rcalib = meta->getRCalib();

	auto adc2fC = [channelCoder, shape](unsigned int adc){
	    float fC = 0;
	    for (auto capId : {0,1,2,3}) fC += channelCoder->charge(*shape, adc, capId);
	    return fC/4;
	};

	int qieType =conditions.getHcalQIEType(cell)->getValue();

	const size_t SIZE = qieType==QIE8 ? INPUT_LUT_SIZE : UPGRADE_LUT_SIZE;
	const int MASK = qieType==QIE8 ? QIE8_LUT_BITMASK :
                         qieType==QIE10 ? QIE10_LUT_BITMASK : QIE11_LUT_BITMASK;
        double linearLSB = linearLSB_QIE8_;
        if (qieType == QIE11 and cell.ietaAbs() == topo_->lastHBRing())
           linearLSB = linearLSB_QIE11Overlap_;
        else if (qieType == QIE11)
           linearLSB = linearLSB_QIE11_;

	lut.resize(SIZE, 0);

	// Input LUT for HB/HE/HF
	if (subdet == HcalBarrel || subdet == HcalEndcap){

	    int granularity = meta->getLutGranularity();

	    double correctionPhaseNS = conditions.getHcalRecoParam(cell)->correctionPhaseNS();
	    for (unsigned int adc = 0; adc < SIZE; ++adc) {
		if (isMasked) lut[adc] = 0;
		else {
		    double nonlinearityCorrection = 1.0;
		    double containmentCorrection2TSCorrected = 1.0;
		    // SiPM nonlinearity was not corrected in 2017
		    // and containment corrections  were not
		    // ET-dependent prior to 2018
		    if(is2018OrLater) {
		      double containmentCorrection1TS = pulseCorr_->correction(cell, 1, correctionPhaseNS, adc2fC(adc));
		      // Use the 1-TS containment correction to estimate the charge of the pulse
		      // from the individual samples
		      double correctedCharge = containmentCorrection1TS*adc2fC(adc);
		      containmentCorrection2TSCorrected = pulseCorr_->correction(cell, 2, correctionPhaseNS, correctedCharge);
		      if(qieType==QIE11) {
			const HcalSiPMParameter& siPMParameter(*conditions.getHcalSiPMParameter(cell));
			HcalSiPMnonlinearity corr(conditions.getHcalSiPMCharacteristics()->getNonLinearities(siPMParameter.getType()));
			const double fcByPE = siPMParameter.getFCByPE();
			const double effectivePixelsFired = correctedCharge/fcByPE;
			nonlinearityCorrection = corr.getRecoCorrectionFactor(effectivePixelsFired);
		      }
		    }
                    if (allLinear_)
		      lut[adc] = (LutElement) std::min(std::max(0, int((adc2fC(adc) - ped) * gain * rcalib * nonlinearityCorrection * containmentCorrection2TSCorrected / linearLSB / cosh_ieta(cell.ietaAbs(), cell.depth(), HcalEndcap))), MASK);
                    else
                       lut[adc] = (LutElement) std::min(std::max(0, int((adc2fC(adc) - ped) * gain * rcalib * nonlinearityCorrection * containmentCorrection2TSCorrected / nominalgain_ / granularity)), MASK);

		    if(qieType==QIE11){
			if (adc >= mipMin and adc < mipMax) lut[adc] |= QIE11_LUT_MSB0;
			else if (adc >= mipMax) lut[adc] |= QIE11_LUT_MSB1;
		    }
		}
	    }
	}  
	else if (subdet == HcalForward){
	    for (unsigned int adc = 0; adc < SIZE; ++adc) {
		if (isMasked) lut[adc] = 0;
		else {
		  lut[adc] = std::min(std::max(0,int((adc2fC(adc) - ped) * gain * rcalib / lsb_ / cosh_ieta_[cell.ietaAbs()])), MASK);
		    if(adc>FG_HF_threshold_) lut[adc] |= QIE10_LUT_MSB;
		}
	    }
        }
    } 
}

void HcaluLUTTPGCoder::adc2Linear(const HBHEDataFrame& df, IntegerCaloSamples& ics) const {
  int lutId = getLUTId(df.id());
  const Lut& lut = inputLUT_.at(lutId);
  for (int i=0; i<df.size(); i++){
    ics[i] = (lut.at(df[i].adc()) & QIE8_LUT_BITMASK);
  }
}

void HcaluLUTTPGCoder::adc2Linear(const HFDataFrame& df, IntegerCaloSamples& ics) const {
  int lutId = getLUTId(df.id());
  const Lut& lut = inputLUT_.at(lutId);
  for (int i=0; i<df.size(); i++){
    ics[i] = (lut.at(df[i].adc()) & QIE8_LUT_BITMASK);
  }
}

void HcaluLUTTPGCoder::adc2Linear(const QIE10DataFrame& df, IntegerCaloSamples& ics) const {
  int lutId = getLUTId(HcalDetId(df.id()));
  const Lut& lut = inputLUT_.at(lutId);
  for (int i=0; i<df.samples(); i++){
    ics[i] = (lut.at(df[i].adc()) & QIE10_LUT_BITMASK);
  }
}

void HcaluLUTTPGCoder::adc2Linear(const QIE11DataFrame& df, IntegerCaloSamples& ics) const {
  int lutId = getLUTId(HcalDetId(df.id()));
  const Lut& lut = inputLUT_.at(lutId);
  for (int i=0; i<df.samples(); i++){
    ics[i] = (lut.at(df[i].adc()) & QIE11_LUT_BITMASK);
  }
}

unsigned short HcaluLUTTPGCoder::adc2Linear(HcalQIESample sample, HcalDetId id) const {
  int lutId = getLUTId(id);
  return ((inputLUT_.at(lutId)).at(sample.adc()) & QIE8_LUT_BITMASK);
}

float HcaluLUTTPGCoder::getLUTPedestal(HcalDetId id) const {
  int lutId = getLUTId(id);
  return ped_.at(lutId);
}

float HcaluLUTTPGCoder::getLUTGain(HcalDetId id) const {
  int lutId = getLUTId(id);
  return gain_.at(lutId);
}

std::vector<unsigned short> HcaluLUTTPGCoder::getLinearizationLUT(HcalDetId id) const{
    int lutId = getLUTId(id);
    return inputLUT_.at(lutId);
}

void HcaluLUTTPGCoder::lookupMSB(const HBHEDataFrame& df, std::vector<bool>& msb) const{
  msb.resize(df.size());
  for (int i=0; i<df.size(); ++i)
    msb[i] = getMSB(df.id(), df.sample(i).adc());
}

bool HcaluLUTTPGCoder::getMSB(const HcalDetId& id, int adc) const{
  int lutId = getLUTId(id);
  const Lut& lut = inputLUT_.at(lutId);
  return (lut.at(adc) & QIE8_LUT_MSB);
}

void HcaluLUTTPGCoder::lookupMSB(const QIE10DataFrame& df, std::vector<bool>& msb) const{
    msb.resize(df.samples());
    int lutId = getLUTId(HcalDetId(df.id()));
    const Lut& lut = inputLUT_.at(lutId);
    for (int i = 0; i < df.samples(); ++i) {
	msb[i] = lut.at(df[i].adc()) & QIE10_LUT_MSB;
    }
}

void
HcaluLUTTPGCoder::lookupMSB(const QIE11DataFrame& df, std::vector<std::bitset<2>>& msb) const
{
   int lutId = getLUTId(HcalDetId(df.id()));
   const Lut& lut = inputLUT_.at(lutId);
   for (int i = 0; i < df.samples(); ++i) {
      msb[i][0] = lut.at(df[i].adc()) & QIE11_LUT_MSB0;
      msb[i][1] = lut.at(df[i].adc()) & QIE11_LUT_MSB1;
   }
}

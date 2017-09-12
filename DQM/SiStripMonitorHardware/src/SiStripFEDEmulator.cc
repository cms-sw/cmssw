#include "DQM/SiStripMonitorHardware/interface/SiStripSpyEventMatcher.h"
#include "DQM/SiStripMonitorHardware/interface/SiStripFEDEmulator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "boost/bind.hpp"

using edm::LogError;
using edm::LogInfo;
using edm::LogWarning;

namespace sistrip{

  const char* FEDEmulator::messageLabel_ = "SiStripFEDEmulator";


  FEDEmulator::FEDEmulator() {
    byModule_ = false;
    minStrip_ = 0;
    maxStrip_ = 0;
    pedestals_.clear();
    noises_.clear();
    medians_.clear();

    detId_ = 0;
    nPairs_ = 0;
    pair_ = 0;

  }



  FEDEmulator::~FEDEmulator() {

  }

  void FEDEmulator::initialise(const bool byModule)
  {
    byModule_ = byModule;
    pedestals_.clear();
    noises_.clear();
    medians_.clear();
  }

  void FEDEmulator::initialiseModule(const uint32_t aDetId,
				     const uint32_t aNPairs,
				     const uint32_t aPair)
  {
    detId_ = aDetId;
    nPairs_ = aNPairs;
    pair_ = aPair;

    minStrip_ = 0;
    maxStrip_ = nPairs_*sistrip::STRIPS_PER_FEDCH;
    if (!byModule_) {
      minStrip_ = sistrip::STRIPS_PER_FEDCH*pair_;
      maxStrip_ = sistrip::STRIPS_PER_FEDCH*(pair_+1);
    }

    //need resize because it will not be filled by push_back...
    pedestals_.clear();
    pedestals_.resize(nPairs_*sistrip::STRIPS_PER_FEDCH);

    noises_.clear();
    noises_.resize(nPairs_*sistrip::STRIPS_PER_FEDCH);

    medians_.clear();
    medians_.reserve(nPairs_*2);

  }

  void FEDEmulator::retrievePedestals(const edm::ESHandle<SiStripPedestals> & aHandle){
    try {
      SiStripPedestals::Range pedestalsRange = aHandle->getRange(detId_);
      aHandle->allPeds(pedestals_,pedestalsRange);
    }
    catch(const cms::Exception& e) {
      LogError(messageLabel_) << " Something wrong whilst getting pedestals: size = " 
			                  << pedestals_.size() << " detId = " 
			                  << detId_ << std::endl << e.what();
      //throw cms::Exception("ProductNotFound") << "Problem with pedestals..." ;
    }
  }//end of retrievePedestals method.

  void FEDEmulator::retrieveNoises(const edm::ESHandle<SiStripNoises> & aHandle){
    try {
      SiStripNoises::Range noisesRange = aHandle->getRange(detId_);
      aHandle->allNoises(noises_,noisesRange);
    }
    catch(const cms::Exception& e) {
      LogError(messageLabel_) << " Something wrong whilst getting noises: size = " 
			                  << noises_.size() << " detId = " 
			                  << detId_ << std::endl << e.what();
      //throw cms::Exception("ProductNotFound") << "Problem with noises..." ;
    }
  }//end of retrieveNoises method.

  void FEDEmulator::subtractPedestals(const edm::DetSetVector<SiStripRawDigi>::const_iterator & inputChannel,
				      std::vector<SiStripRawDigi> & pedsDetSetData,
				      std::vector<SiStripProcessedRawDigi> & noiseDetSetData,
				      std::vector<SiStripRawDigi> & pedSubtrDetSetData,
				      std::vector<uint32_t> & medsDetSetData,
				      const bool fillApvsForCM)
  {

    edm::DetSet<SiStripRawDigi>::const_iterator digi_it = inputChannel->begin();
    ////for median calculation  
    uint32_t lCount = 0;
    std::vector<std::vector<uint16_t> > apvs;
    apvs.reserve(nPairs_*2);

    //subtract pedestals and fill apv vector 
    //for median calculation
    //in a module, 256*nPairs, but here we are dealing
    //channel by channel, so 256 values at a time in the pedestal object.
    //reordered = 0-127 APV0 128-255 APV1

    std::vector<uint16_t> apvVec;;
    apvVec.reserve(sistrip::STRIPS_PER_APV);
      
    for (uint32_t st = minStrip_; st < maxStrip_; st++) {
      uint16_t ped = static_cast<uint16_t>(pedestals_[st]);
      pedsDetSetData.push_back(SiStripRawDigi(ped));

      float noise = noises_[st];
      noiseDetSetData.push_back(SiStripProcessedRawDigi(noise));
    
      if (digi_it == inputChannel->end()) {
	LogError(messageLabel_) << " Error, end of inputchannel reached for detid " << detId_ << "! Processed " << lCount << " strips." << std::endl;
	break;
      }
    
      int lVal = digi_it->adc() - ped;
      if (lVal < 0) {
	if (digi_it->adc()>0) LogDebug(messageLabel_) << " Negative value after pedestal subtraction ! Detid = " << detId_ << ", strip " << st << ", digi = " << digi_it->adc() << ", ped = " << ped << std::endl;
	lVal = 0;
      }
      pedSubtrDetSetData.push_back( SiStripRawDigi(static_cast<uint16_t>(lVal)) );

      
      if (fillApvsForCM) {
	apvVec.push_back( static_cast<uint16_t>(lVal) );
      }
      
      lCount++;
      ++digi_it;
      
      if (lCount%sistrip::STRIPS_PER_APV == 0 && fillApvsForCM) {
	if (!apvVec.empty()) apvs.push_back(apvVec);
	apvVec.clear();
	apvVec.reserve(sistrip::STRIPS_PER_APV);
      }
      
    }//end of loop over strips
    
    if (fillApvsForCM){
      if (apvs.size() != nPairs_*2) {
	LogError(messageLabel_) << " Error, wrong output size for median calculation for detid " << detId_ << "! Processed " << apvs.size() << " APVs." << std::endl;
      }

      //calculate common mode values
      for (uint32_t iapv(0); iapv<apvs.size(); iapv++){
	std::vector<uint16_t> lVec = apvs[iapv];

	std::vector<uint16_t>::iterator mid = lVec.begin() + (lVec.size()/2 - 1);
	std::nth_element(lVec.begin(), mid, lVec.end());

	//std::sort(lVec.begin(),lVec.end());
	//get the 64th element
	//if (*mid != lVec.at(63)) exit(0);
	//medians_.push_back(lVec.at(63));
	//medsDetSetData.push_back(lVec.at(63));
	medians_.push_back(*mid);
	medsDetSetData.push_back(*mid);
    
      }
    }

  }//end of FEDEmulator::subtractPedestals method.


  void FEDEmulator::subtractCM(const std::vector<SiStripRawDigi> & pedSubtrDetSetData,
			       std::vector<SiStripRawDigi> & cmSubtrDetSetData)
  {
    //subtract Medians
    std::vector<SiStripRawDigi>::const_iterator lDigi = pedSubtrDetSetData.begin();
    uint32_t lCount = 0;

    for (uint32_t st = minStrip_; st < maxStrip_; st++) {
      uint32_t lApvNum = static_cast<uint32_t>(lCount*1./sistrip::STRIPS_PER_APV);
      
      int value = lDigi->adc() - medians_.at(lApvNum);
      if (value < 0) {
	//FED doesn't handle negative values
	value = 0;
      }
      cmSubtrDetSetData.push_back(SiStripRawDigi(static_cast<uint16_t>(value)));

      ++lDigi;
      lCount++;
    }

  }

  void FEDEmulator::zeroSuppress(const std::vector<SiStripRawDigi> & cmSubtrDetSetData,
				 edm::DetSet<SiStripDigi>    & zsDetSetData,
				 const std::unique_ptr<SiStripRawProcessingAlgorithms> & algorithms)
  {
    //transform the input digis to a vector of integers
    std::vector<int16_t> cmSubtrRawDigis;
    transform(cmSubtrDetSetData.begin(), 
	      cmSubtrDetSetData.end(),
	      back_inserter(cmSubtrRawDigis),
	      boost::bind(&SiStripRawDigi::adc ,_1)
	      );
    algorithms->suppressor->suppress(cmSubtrRawDigis, 0,
				     zsDetSetData);
    
  }//end of FEDEmulator::zeroSuppress method.


  
  void FEDEmulator::fillPeds(const edm::DetSetVector<SiStripRawDigi>::const_iterator & peds){
    for (edm::DetSet<SiStripRawDigi>::const_iterator iApv = peds->begin(); 
	 iApv != peds->end(); ++iApv) {
      pedestals_.push_back(iApv->adc());
    }
  }
   
  void FEDEmulator::fillNoises(const edm::DetSetVector<SiStripProcessedRawDigi>::const_iterator & noises){
    for (edm::DetSet<SiStripProcessedRawDigi>::const_iterator iApv = noises->begin(); 
	 iApv != noises->end(); ++iApv) {
      noises_.push_back(iApv->adc());
    }
  }
  
  void FEDEmulator::fillMedians(const std::map<uint32_t,std::vector<uint32_t> >::const_iterator & meds){
    std::vector<uint32_t>::const_iterator iApv = (meds->second).begin();
    for ( ; iApv != (meds->second).end(); ++iApv) {
      medians_.push_back(*iApv);
    }
  }

  void FEDEmulator::print(std::ostream & aOs) {
    aOs << "===============================" << std::endl
	<< " ===== FEDEmulator::print =====" << std::endl
	<< " === byModule = " << byModule_ << std::endl
	<< " === minStrip = " << minStrip_ << std::endl
	<< " === maxStrip = " << maxStrip_ << std::endl
	<< " === size of pedestals = " << pedestals_.size() << std::endl
	<< " === size of noises = " << noises_.size() << std::endl
	<< " === size of medians = " << medians_.size() << std::endl
	<< " === detId = " << detId_ << std::endl
	<< " === nPairs = " << nPairs_ << std::endl
	<< " === pair = " << pair_ << std::endl
	<< "===============================" << std::endl;
  }


  void FEDEmulator::printPeds(std::ostream & aOs) {
    aOs << "=========================================" << std::endl
	<< " ===== FEDEmulator::printPedestals =====" << std::endl;
    for (unsigned int i(0); i<pedestals_.size(); i++){
      if (i%32 == 0) aOs << std::endl;
      aOs << pedestals_[i] << " " ;
    }
    aOs << "=========================================" << std::endl;
  }

  void FEDEmulator::printNoises(std::ostream & aOs) {
    aOs << "=========================================" << std::endl
	<< " ===== FEDEmulator::printNoises =====" << std::endl;
    for (unsigned int i(0); i<noises_.size(); i++){
      if (i%32 == 0) aOs << std::endl;
      aOs << noises_[i] << " " ;
    }
    aOs << "=========================================" << std::endl;
  }

  void FEDEmulator::printMeds(std::ostream & aOs) {
    aOs << "=========================================" << std::endl
	<< " ===== FEDEmulator::printMedians =====" << std::endl;
    for (unsigned int i(0); i<medians_.size(); i++){
      if (i%32 == 0) aOs << std::endl;
      aOs << medians_[i] << " " ;
    }
    aOs << "=========================================" << std::endl;
  }




}//namespace sistrip

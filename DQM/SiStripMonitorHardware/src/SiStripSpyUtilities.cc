#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

// Needed for the FED cabling and pedestals
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CondFormats/DataRecord/interface/SiStripPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"

#include "DQM/SiStripMonitorHardware/interface/SiStripFEDSpyBuffer.h"
#include "DQM/SiStripMonitorHardware/interface/SiStripSpyUtilities.h"

using edm::LogError;
using edm::LogWarning;
using edm::LogInfo;


namespace sistrip {
  SpyUtilities::SpyUtilities() :
    cabling_(nullptr),
    cacheId_(0),
    detCabling_(nullptr),
    cacheIdDet_(0),
    pedsCacheId_(0),
    pedsHandle_(nullptr),
    noiseCacheId_(0),
    noiseHandle_(nullptr)
  {
    
  }
  
  SpyUtilities::~SpyUtilities()
  {
    if ( cabling_ ) cabling_ = nullptr;
    if ( detCabling_ ) detCabling_ = nullptr;
  }

  const SiStripFedCabling*  SpyUtilities::getCabling( const edm::EventSetup& setup )
  {
    
    uint32_t cache_id = setup.get<SiStripFedCablingRcd>().cacheIdentifier();
   
    if ( cacheId_ != cache_id ) { // If the cache ID has changed since the last update...
      // Update the cabling object
      edm::ESHandle<SiStripFedCabling> c;
      setup.get<SiStripFedCablingRcd>().get( c );
      cabling_ = c.product();
      
//       	  if ( edm::isDebugEnabled() ) {
// 	    if ( !cacheId_ ) { // First time cabling has been retrieved - print it out in full.
// 	      std::stringstream ss;
// 	      ss << "[sistrip::SpyChannelUnpackerModule::" << __func__ << "]"
// 		 << " Updating cabling for first time..." << std::endl
// 		 << " Terse print out of FED cabling:" << std::endl;
// 	      //cabling_->terse(ss);
// 	      //LogTrace("SiStripMonitorHardwareUnpacker") << ss.str();
// 	    } // end of cacheId_ check
// 	  } // end of debugEnabled check
      
//       if ( edm::isDebugEnabled() ) {
// 	std::stringstream sss;
// 	sss << "[sistrip::SpyUtilities::" << __func__ << "]"
// 	    << " Summary of FED cabling:" << std::endl;
// 	cabling_->summary(sss);
// 	LogTrace("SiStripSpyUtilities") << sss.str();
//       }

      // Update the cache ID with the new value.
      cacheId_ = cache_id;

    } // end of new cache ID check
        
    return cabling_;
  }

  const SiStripDetCabling*  SpyUtilities::getDetCabling( const edm::EventSetup& setup )
  {
    
    uint32_t cache_id = setup.get<SiStripDetCablingRcd>().cacheIdentifier();//.get( cabling_ );
   
    if ( cacheIdDet_ != cache_id ) { // If the cache ID has changed since the last update...
      // Update the cabling object
      edm::ESHandle<SiStripDetCabling> c;
      setup.get<SiStripDetCablingRcd>().get( c );
      detCabling_ = c.product();
      cacheIdDet_ = cache_id;
    } // end of new cache ID check
        
    return detCabling_;
  }

  edm::ESHandle<SiStripPedestals> SpyUtilities::getPedestalHandle(const edm::EventSetup& eventSetup)
  {
    //check if new pedestal values are available
    uint32_t lCacheId = eventSetup.get<SiStripPedestalsRcd>().cacheIdentifier();
    if (lCacheId != pedsCacheId_) {
      eventSetup.get<SiStripPedestalsRcd>().get(pedsHandle_);
      pedsCacheId_ = lCacheId;
    }

    return pedsHandle_;
  }


  edm::ESHandle<SiStripNoises> SpyUtilities::getNoiseHandle(const edm::EventSetup& eventSetup)
  {
    //check if new noise values are available
    uint32_t lCacheId = eventSetup.get<SiStripNoisesRcd>().cacheIdentifier();
    if (lCacheId != noiseCacheId_) {
      eventSetup.get<SiStripNoisesRcd>().get(noiseHandle_);
      noiseCacheId_ = lCacheId;
    }

    return noiseHandle_;
  }


  const SpyUtilities::Frame 
  SpyUtilities::extractFrameInfo(const edm::DetSetVector<SiStripRawDigi>::detset & channelDigis,
				 bool aPrintDebug)
  {

    SpyUtilities::Frame lFrame;
    lFrame.detId = channelDigis.detId();
    lFrame.firstHeaderBit = 0;
    lFrame.firstTrailerBit = 0;
    lFrame.digitalLow = 0;
    lFrame.digitalHigh = 0;
    lFrame.baseline = 0;
    lFrame.apvErrorBit.first = false;
    lFrame.apvErrorBit.second = false;
    lFrame.apvAddress.first = 0;
    lFrame.apvAddress.second = 0;

    uint16_t min = 0x3FF;
    uint16_t max = 0;
    edm::DetSetVector<SiStripRawDigi>::detset::const_iterator iDigi = channelDigis.begin();
    const edm::DetSetVector<SiStripRawDigi>::detset::const_iterator endChannelDigis = channelDigis.end();

    //counters for outputting warnings
    uint16_t numzeroes = 0, numsats = 0;

    if (iDigi == endChannelDigis) return lFrame;

    for (; iDigi != endChannelDigis; ++iDigi) {
      const uint16_t val = iDigi->adc();
      if (val < min) min = val;
      if (val > max) max = val;
      if (val==0)     numzeroes++;
      if (val==0x3FF) numsats++;
      lFrame.baseline += val;
    }

    if (!channelDigis.empty()) lFrame.baseline = lFrame.baseline/channelDigis.size();
    lFrame.digitalLow = min;
    lFrame.digitalHigh = max;

    const uint16_t threshold = static_cast<uint16_t>( (2.0 * static_cast<double>(max-min)) / 3.0 );

    if (aPrintDebug){
//       if ( edm::isDebugEnabled() ) {
// 	LogDebug("SiStripSpyUtilities") << "Channel with key: " << lFrame.detId
// 					<< " Min: " << min << " Max: " << max
// 					<< " Range: " << (max-min) << " Threshold: " << threshold;
//       }
      if (numzeroes>0 || numsats>0) {
	edm::LogWarning("SiStripSpyUtilities") << "Channel with key: " << lFrame.detId << " has "
					       << numzeroes << " zero and "
					       << numsats   << " saturated samples.";
      }
    }

    lFrame.firstHeaderBit = findHeaderBits(channelDigis,threshold);
    lFrame.firstTrailerBit = findTrailerBits(channelDigis,threshold);

    lFrame.apvErrorBit = findAPVErrorBits(channelDigis,threshold,lFrame.firstHeaderBit);
    lFrame.apvAddress = findAPVAddresses(channelDigis,threshold,lFrame.firstHeaderBit);
  
    return lFrame;
 
  }

  const uint16_t SpyUtilities::range(const SpyUtilities::Frame & aFrame)
  {
    if (aFrame.digitalHigh < aFrame.digitalLow) return 0;
    else return aFrame.digitalHigh-aFrame.digitalLow;
  }

  const uint16_t SpyUtilities::threshold(const SpyUtilities::Frame & aFrame)
  {
    return static_cast<uint16_t>( (2.0 * static_cast<double>(range(aFrame))) / 3.0 );
  }

  const uint8_t SpyUtilities::extractAPVaddress(const SpyUtilities::Frame & aFrame)
  {

    if (aFrame.apvErrorBit.first == false) return aFrame.apvAddress.first;
    else if (aFrame.apvErrorBit.second == false) {
      return aFrame.apvAddress.second;
    }
    else {
      return 0;
    }
    
  }
   
  void SpyUtilities::getMajorityHeader(const edm::DetSetVector<SiStripRawDigi> *aInputDigis,
				       uint16_t & aFirstHeaderBit,
				       bool printResult)
  {

    std::vector<uint16_t> lFirstBitVec;
    lFirstBitVec.reserve(aInputDigis->size());
    aFirstHeaderBit = 0;
    edm::DetSetVector<SiStripRawDigi>::const_iterator lDigis = aInputDigis->begin();
    
    for ( ; lDigis != aInputDigis->end(); lDigis++){
      sistrip::SpyUtilities::Frame lFrame = sistrip::SpyUtilities::extractFrameInfo(*lDigis);
      lFirstBitVec.push_back(lFrame.firstHeaderBit);
    }

    std::pair<uint16_t,uint32_t> lMaj = sistrip::SpyUtilities::findMajorityValue(lFirstBitVec);
    aFirstHeaderBit = lMaj.first;
    uint32_t lMajorityCounter = lMaj.second;
    
    //header is 24-sample long (2*8+2+6)
    uint16_t lFirstTrailerBit = aFirstHeaderBit+24+sistrip::STRIPS_PER_FEDCH;
    
    if (printResult)
      {
	LogInfo("SiStripSpyUtilities") << " -- Found majority position of first header (trailer) bit: " 
				       << aFirstHeaderBit
				       << " (" << lFirstTrailerBit 
				       << ") for " << lMajorityCounter << " out of " << lFirstBitVec.size() << " channels."
				       << std::endl;
      }
  }


  const bool SpyUtilities::isValid(const SpyUtilities::Frame & aFrame,
				   const FrameQuality & aQuality,
				   const uint16_t aExpectedPos)
  {

    uint16_t lRange = sistrip::SpyUtilities::range(aFrame);

    if (lRange < aQuality.minDigiRange || lRange > aQuality.maxDigiRange) {
      return false;
    }
    else if (aFrame.digitalLow < aQuality.minZeroLight || aFrame.digitalLow > aQuality.maxZeroLight) {
      return false;
    }
    else if (aFrame.digitalHigh < aQuality.minTickHeight || aFrame.digitalHigh > aQuality.maxTickHeight){
      return false;
    }
    //if expectedPos=0: return true whatever the position of header is...
    else if ( aExpectedPos > 0 && 
	      (
	       !(aFrame.firstHeaderBit == aExpectedPos && 
		 aFrame.firstTrailerBit == aExpectedPos+24+sistrip::STRIPS_PER_FEDCH)
	       )
	      ) {
      return false;
    }
    else if (aFrame.apvErrorBit.first && aFrame.apvErrorBit.second) {
      return false;
    }
    
    return true;
  }


  const uint16_t SpyUtilities::findHeaderBits(const edm::DetSetVector<SiStripRawDigi>::detset & channelDigis,
					      const uint16_t threshold)
  {

    // Loop over digis looking for first above threshold
    uint8_t aboveThreshold = 0;
    bool foundHeader = false;
    uint16_t count = 0;

    edm::DetSetVector<SiStripRawDigi>::detset::const_iterator iDigi = channelDigis.begin();
    const edm::DetSetVector<SiStripRawDigi>::detset::const_iterator endChannelDigis = channelDigis.end();

    for (; iDigi != endChannelDigis; ++iDigi) {
      if ( iDigi->adc() > threshold) {
	aboveThreshold++;
      }
      else {
	aboveThreshold = 0;
      }
      if (aboveThreshold == 6) {foundHeader = true; break; }
      count++;
    }//end of loop over digis

    //break before incrementing the last time... so count-5 is the first header sample.
    if (foundHeader && count < 5) return 0;
    if (foundHeader) return count-5;
    return sistrip::SPY_SAMPLES_PER_CHANNEL;
 
  }

  const uint16_t SpyUtilities::findTrailerBits(const edm::DetSetVector<SiStripRawDigi>::detset & channelDigis,
					       const uint16_t threshold)
  {

    // Loop over digis looking for last above threshold
    uint8_t aboveThreshold = 0;
    bool foundTrailer = false;

    //discard the first 30 values, which will have some digital high in them...
    //start searching from the expected position : sometimes after 24+256 samples,
    //normally at 6+24+256 if 6-bit low before tickmark header bits...
    uint16_t count = 24+sistrip::STRIPS_PER_FEDCH;

    if (count >= sistrip::SPY_SAMPLES_PER_CHANNEL) return sistrip::SPY_SAMPLES_PER_CHANNEL;

    edm::DetSetVector<SiStripRawDigi>::detset::const_iterator iDigi = channelDigis.begin()+count;
    const edm::DetSetVector<SiStripRawDigi>::detset::const_iterator endChannelDigis = channelDigis.end();

    for (; iDigi != endChannelDigis; ++iDigi) {
      if ( iDigi->adc() > threshold) {
	aboveThreshold++;
      }
      else {
	aboveThreshold = 0;
      }
      if (aboveThreshold == 2) {foundTrailer = true; break; }
      count++;
    }//end of loop over digis

    //break before incrementing the last time... so count-1 is the first trailer sample.
    if (foundTrailer && count < 1) return 0;
    if (foundTrailer) return count-1;
    return sistrip::SPY_SAMPLES_PER_CHANNEL;
 
  }



  const std::pair<bool,bool> 
  SpyUtilities::findAPVErrorBits(const edm::DetSetVector<SiStripRawDigi>::detset & channelDigis,
				 const uint16_t threshold,
				 const uint16_t aFirstBits)
  {
  
    // Loop over digis looking for firstHeader+6+16
    uint16_t count = aFirstBits+22;

    std::pair<bool,bool> lPair = std::pair<bool,bool>(false,false);

    //if header invalid: we don't know what apverr is....
    if (count >= sistrip::SPY_SAMPLES_PER_CHANNEL-1) return lPair;

    edm::DetSetVector<SiStripRawDigi>::detset::const_iterator iDigi = channelDigis.begin()+count;
    const edm::DetSetVector<SiStripRawDigi>::detset::const_iterator endChannelDigis = channelDigis.end();

    //double check....
    if (iDigi == endChannelDigis) return lPair;

    if ( iDigi->adc() <= threshold) lPair.first = true;
    ++iDigi;

    //triple check...
    if (iDigi == endChannelDigis) return std::pair<bool,bool>(false,false);

    if ( iDigi->adc() <= threshold) lPair.second = true;

    return lPair; 
  }


  const std::pair<uint8_t,uint8_t> 
  SpyUtilities::findAPVAddresses(const edm::DetSetVector<SiStripRawDigi>::detset & channelDigis,
				 const uint16_t threshold,
				 const uint16_t aFirstBits)
  {
  
    // Loop over digis looking for firstHeader+6
    uint16_t count = aFirstBits+6;
    std::pair<uint8_t,uint8_t> lPair = std::pair<uint8_t,uint8_t>(0,0);

    //check enough room to have 16 values....
    if (count >= sistrip::SPY_SAMPLES_PER_CHANNEL-15) return lPair;

    edm::DetSetVector<SiStripRawDigi>::detset::const_iterator iDigi = channelDigis.begin()+count;
    const edm::DetSetVector<SiStripRawDigi>::detset::const_iterator endChannelDigis = channelDigis.end();

    //double check....
    if (iDigi == endChannelDigis) return lPair;

    for (uint8_t i = 0; i < 16; ++i) {
      if ( iDigi->adc() > threshold) {
	//data is MSB first
	if (i%2==0)
	  lPair.first |= (0x80 >> static_cast<uint8_t>(i/2));
	else 
	  lPair.second |= (0x80 >> static_cast<uint8_t>(i/2));
      }
      ++iDigi;
    }

    return lPair;

  }


  std::string SpyUtilities::print(const SpyUtilities::Frame & aFrame,
				  std::string aErr)
  {

    std::ostringstream lOs;
    lOs << " ------------------------------------------------------" << std::endl
	<< " -- Error: " << aErr << std::endl
	<< " ------- Printing Frame for detId " << aFrame.detId << " --------" << std::endl
	<< " -- firstHeaderBit = " << aFrame.firstHeaderBit << std::endl
	<< " -- firstTrailerBit = " << aFrame.firstTrailerBit << std::endl
	<< " -- digitalLow = " << aFrame.digitalLow << std::endl
	<< " -- digitalHigh = " << aFrame.digitalHigh << std::endl
	<< " -- baseline = " << aFrame.baseline << std::endl
	<< " -- apvErrorBits = " << aFrame.apvErrorBit.first
	<< " " << aFrame.apvErrorBit.second << std::endl
	<< " -- apvAddresses = " << static_cast<uint16_t>(aFrame.apvAddress.first)
	<< " " << static_cast<uint16_t>(aFrame.apvAddress.second) << std::endl
	<< " ------------------------------------------------------" << std::endl;
    return lOs.str();

  }


  void SpyUtilities::fedIndex(const uint32_t aFedIndex,
			      uint16_t & aFedId,
			      uint16_t & aFedChannel){

    //find the corresponding detId (for the pedestals)
    aFedId = static_cast<uint16_t>(aFedIndex/sistrip::FEDCH_PER_FED);
    aFedChannel = static_cast<uint16_t>(aFedIndex%sistrip::FEDCH_PER_FED);
  
    if (aFedId < sistrip::FED_ID_MIN ||
	aFedId > sistrip::FED_ID_MAX ||
	aFedChannel >= sistrip::FEDCH_PER_FED ) { 
      aFedId = sistrip::invalid_;
      aFedChannel = sistrip::invalid_;
    }

  }

  std::pair<uint16_t,uint32_t> SpyUtilities::findMajorityValue(std::vector<uint16_t>& values,
							       const uint16_t aFedId)
  {

    uint32_t lTot = values.size();
    if (!lTot) return std::pair<uint16_t,uint32_t>(0,0);

    std::sort(values.begin(),values.end());
    uint32_t lMajorityCounter = 0;
    uint16_t lMaj = 0;

    std::vector<uint16_t>::iterator lIter = values.begin();
    for ( ; lIter != values.end(); ) {
      uint32_t lCounter = std::count(lIter,values.end(),*lIter);
      if (lCounter > lMajorityCounter) {
	lMajorityCounter = lCounter;
	lMaj = *lIter;
      }
      lIter += lCounter;
    }
    
    //std::cout << " -- Found majority value " << lMaj << " for " << lMajorityCounter << " elements out of " << values.size() << "." << std::endl;

    if (static_cast<float>(lMajorityCounter)/lTot < 0.5) {
      LogError("SiStripSpyUtilities") << " -- Found majority position for index "
				      << aFedId
				      << ": " << lMaj
				      << " for less than half the values : " << lMajorityCounter << " out of " << lTot << " values."
				      << std::endl;
    }

    return std::pair<uint16_t,uint32_t>(lMaj,lMajorityCounter);
    
  }

  void SpyUtilities::fillFEDMajorities(const std::map<uint32_t,uint32_t>& channelValues, 
				       std::vector<uint32_t> & fedMajoritiesToFill)
  {

    std::map<uint32_t,uint32_t>::const_iterator lMapIter = channelValues.begin();
    uint16_t lPreviousFedId = 0;
    std::vector<uint16_t> lAddrVec;
    lAddrVec.reserve(sistrip::FEDCH_PER_FED);
    fedMajoritiesToFill.resize(sistrip::FED_ID_MAX-sistrip::FED_ID_MIN+1,0);
    uint32_t lChCount = 0;

    for ( ; lMapIter != channelValues.end(); ++lMapIter,++lChCount){

      uint16_t lFedId = static_cast<uint16_t>(lMapIter->first/sistrip::FEDCH_PER_FED);

      if (lPreviousFedId == 0) {
	lPreviousFedId = lFedId;
      }
      if (lFedId == lPreviousFedId) {
	lAddrVec.push_back(lMapIter->second);
      }
      if (lFedId != lPreviousFedId || (lChCount == channelValues.size()-1)) {
	//extract majority address

	uint32_t lMaj = sistrip::SpyUtilities::findMajorityValue(lAddrVec,lPreviousFedId).first;
	fedMajoritiesToFill[lPreviousFedId] = lMaj;

	lAddrVec.clear();
	
	//if new fed, fill the first channel
	if (lFedId != lPreviousFedId) {
	  lAddrVec.push_back(lMapIter->second);
	  lPreviousFedId = lFedId;
	}

      }
    }

  }

}//namespace

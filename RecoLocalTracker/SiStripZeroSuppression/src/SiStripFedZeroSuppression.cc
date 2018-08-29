#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripFedZeroSuppression.h"

#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/DataRecord/interface/SiStripThresholdRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripThreshold.h"

//#define DEBUG_SiStripZeroSuppression_
//#define ML_DEBUG 
using namespace std;

void SiStripFedZeroSuppression::init(const edm::EventSetup& es){
  uint32_t n_cache_id = es.get<SiStripNoisesRcd>().cacheIdentifier();
  uint32_t t_cache_id = es.get<SiStripThresholdRcd>().cacheIdentifier();

  if(n_cache_id != noise_cache_id) {
    es.get<SiStripNoisesRcd>().get( noiseHandle );
    noise_cache_id = n_cache_id;
  }
  if(t_cache_id != threshold_cache_id) {
    es.get<SiStripThresholdRcd>().get( thresholdHandle );
    threshold_cache_id = t_cache_id;
  }
}

void SiStripFedZeroSuppression::suppress(const std::vector<SiStripDigi>& in, std::vector<SiStripDigi>& selectedSignal, uint32_t detID){
  suppress(in, selectedSignal, detID, noiseHandle, thresholdHandle);
}

void SiStripFedZeroSuppression::suppress(const std::vector<SiStripDigi>& in, std::vector<SiStripDigi>& selectedSignal, uint32_t detID,
							edm::ESHandle<SiStripNoises> & noiseHandle,edm::ESHandle<SiStripThreshold> & thresholdHandle){

  int i;  
  int inSize = in.size();
  SiStripNoises::Range detNoiseRange = noiseHandle->getRange(detID);
  SiStripThreshold::Range detThRange = thresholdHandle->getRange(detID);

  // reserving more than needed, but quicker than one at a time
  selectedSignal.clear();
  selectedSignal.reserve(inSize);
  for (i = 0; i < inSize; i++) {
    //Find adc values for neighbouring strips
    const uint32_t strip = (uint32_t) in[i].strip();
    
    adc   = in[i].adc();

    SiStripThreshold::Data thresholds=thresholdHandle->getData(strip,detThRange);
    theFEDlowThresh  = static_cast<int16_t>(thresholds.getLth()*noiseHandle->getNoiseFast(strip,detNoiseRange)+0.5);
    theFEDhighThresh = static_cast<int16_t>(thresholds.getHth()*noiseHandle->getNoiseFast(strip,detNoiseRange)+0.5);

    adcPrev  = -9999;
    adcNext  = -9999;
    adcPrev2 = -9999;
    adcNext2 = -9999;
    
    /*
      Since we are not running on 
      Raw data we need to initialize
      all the FED threshold
    */
    
    theNextFEDlowThresh  = theFEDlowThresh;
    theNext2FEDlowThresh = theFEDlowThresh;
    thePrevFEDlowThresh  = theFEDlowThresh;
    thePrev2FEDlowThresh = theFEDlowThresh;
    theNeighFEDlowThresh = theFEDlowThresh;

    theNextFEDhighThresh  = theFEDhighThresh;
    thePrevFEDhighThresh  = theFEDhighThresh;
    theNeighFEDhighThresh = theFEDhighThresh;

    if ( ((strip)%128) == 127){ 
      adcNext = 0;
      theNextFEDlowThresh  = 9999;
      theNextFEDhighThresh = 9999;
    }else if (i + 1 < inSize && in[i+1].strip() == strip + 1) {
      adcNext = in[i+1].adc();
      SiStripThreshold::Data thresholds_1=thresholdHandle->getData(strip+1,detThRange);
      theNextFEDlowThresh  = static_cast<int16_t>(thresholds_1.getLth()*noiseHandle->getNoiseFast(strip+1,detNoiseRange)+0.5);
      theNextFEDhighThresh = static_cast<int16_t>(thresholds_1.getHth()*noiseHandle->getNoiseFast(strip+1,detNoiseRange)+0.5);
      if ( ((strip)%128) == 126){ 
	adcNext2 = 0;
	theNext2FEDlowThresh  = 9999;
      }else if (i + 2 < inSize && in[i+2].strip() == strip + 2) {
	adcNext2 = in[i+2].adc();
	theNext2FEDlowThresh  = static_cast<int16_t>(thresholdHandle->getData(strip+2,detThRange).getLth()*noiseHandle->getNoiseFast(strip+2,detNoiseRange)+0.5);
      }
    }

    if ( ((strip)%128) == 0){   
      adcPrev = 0;
      thePrevFEDlowThresh  = 9999;
      thePrevFEDhighThresh = 9999;
    }else if (i - 1 >= 0 && in[i-1].strip() == strip - 1) {
      adcPrev = in[i-1].adc();
      SiStripThreshold::Data thresholds_1=thresholdHandle->getData(strip-1,detThRange);
      thePrevFEDlowThresh  = static_cast<int16_t>(thresholds_1.getLth()*noiseHandle->getNoiseFast(strip-1,detNoiseRange)+0.5);
      thePrevFEDhighThresh = static_cast<int16_t>(thresholds_1.getHth()*noiseHandle->getNoiseFast(strip-1,detNoiseRange)+0.5);
      if ( ((strip)%128) == 1){
	adcPrev2 = 0; 
	thePrev2FEDlowThresh  = 9999;
      }else if (i - 2 >= 0 && in[i-2].strip() == strip - 2) {
	adcPrev2 = in[i-2].adc();
	thePrev2FEDlowThresh  = static_cast<int16_t>(thresholdHandle->getData(strip-2,detThRange).getLth()*noiseHandle->getNoiseFast(strip-2,detNoiseRange)+0.5);
      }
    }

    if ( adcNext <= adcPrev){
      adcMaxNeigh = adcPrev;
      theNeighFEDlowThresh  = thePrevFEDlowThresh;
      theNeighFEDhighThresh = thePrevFEDhighThresh;
    } else {
      adcMaxNeigh = adcNext;
      theNeighFEDlowThresh  = theNextFEDlowThresh;
      theNeighFEDhighThresh = theNextFEDhighThresh;
    }
    
    if (isAValidDigi()){
      selectedSignal.push_back(SiStripDigi(strip, adc));
    }
  }
}

void SiStripFedZeroSuppression::suppress(const edm::DetSet<SiStripRawDigi>& in, edm::DetSet<SiStripDigi>& out)
{
  const uint32_t detID = out.id;
  SiStripNoises::Range detNoiseRange = noiseHandle->getRange(detID);
  SiStripThreshold::Range detThRange = thresholdHandle->getRange(detID);
#ifdef DEBUG_SiStripZeroSuppression_
  if (edm::isDebugEnabled())
    LogTrace("SiStripZeroSuppression") << "[SiStripFedZeroSuppression::suppress] Zero suppression on edm::DetSet<SiStripRawDigi>: detID " << detID << " size = " << in.data.size();
#endif
  edm::DetSet<SiStripRawDigi>::const_iterator in_iter=in.data.begin();
  for (;in_iter!=in.data.end();in_iter++){

    const uint32_t strip = (uint32_t) (in_iter-in.data.begin());

#ifdef DEBUG_SiStripZeroSuppression_
    if (edm::isDebugEnabled())
      LogTrace("SiStripZeroSuppression") << "[SiStripFedZeroSuppression::suppress] detID= " << detID << " strip= " <<  strip << "  adc= " << in_iter->adc();
#endif    
    adc   = in_iter->adc();

    SiStripThreshold::Data thresholds=thresholdHandle->getData(strip,detThRange);
    theFEDlowThresh  = static_cast<int16_t>(thresholds.getLth()*noiseHandle->getNoiseFast(strip,detNoiseRange)+0.5);
    theFEDhighThresh = static_cast<int16_t>(thresholds.getHth()*noiseHandle->getNoiseFast(strip,detNoiseRange)+0.5);

    adcPrev = -9999;
    adcNext = -9999;
    /*
      If a strip is the last one on the chip
      set its next neighbor's thresholds to infinity
      because the FED does not merge clusters across
      chip boundaries right now
    */
    if ( strip%128 == 127 ) { 
      adcNext = 0;      
      theNextFEDlowThresh  = 9999;
      theNextFEDhighThresh = 9999;
    }
    else {
      adcNext = (in_iter+1)->adc();
      SiStripThreshold::Data thresholds_1=thresholdHandle->getData(strip+1,detThRange);
      theNextFEDlowThresh  = static_cast<int16_t>(thresholds_1.getLth()*noiseHandle->getNoiseFast(strip+1,detNoiseRange)+0.5);
      theNextFEDhighThresh = static_cast<int16_t>(thresholds_1.getHth()*noiseHandle->getNoiseFast(strip+1,detNoiseRange)+0.5);
    }
    /*
      Similarily, for the first strip 
      on a chip
    */
    if ( strip%128 == 0 ) {
      adcPrev = 0;
      thePrevFEDlowThresh  = 9999;
      thePrevFEDhighThresh = 9999;   
    }
    else {
      adcPrev = (in_iter-1)->adc();
      SiStripThreshold::Data thresholds_1=thresholdHandle->getData(strip-1,detThRange);
      thePrevFEDlowThresh  = static_cast<int16_t>(thresholds_1.getLth()*noiseHandle->getNoiseFast(strip-1,detNoiseRange)+0.5);
      thePrevFEDhighThresh = static_cast<int16_t>(thresholds_1.getHth()*noiseHandle->getNoiseFast(strip-1,detNoiseRange)+0.5);
    }
    if ( adcNext < adcPrev){
      adcMaxNeigh = adcPrev;
      theNeighFEDlowThresh  = thePrevFEDlowThresh;
      theNeighFEDhighThresh = thePrevFEDhighThresh;
    } else {
      adcMaxNeigh = adcNext;
      theNeighFEDlowThresh  = theNextFEDlowThresh;
      theNeighFEDhighThresh = theNextFEDhighThresh;
    }
    
    //Find adc values for next neighbouring strips
    adcPrev2 = -9999;
    adcNext2 = -9999;
    thePrev2FEDlowThresh  = 1;
    theNext2FEDlowThresh  = 1;
    if ( strip%128 >= 126 ) {
      adcNext2 = 0;
      theNext2FEDlowThresh  = 9999;
    }
    else if ( strip%128 < 126 ) {
      adcNext2 = (in_iter+2)->adc();
     theNext2FEDlowThresh  = static_cast<int16_t>(thresholdHandle->getData(strip+2,detThRange).getLth()*noiseHandle->getNoiseFast(strip+2,detNoiseRange)+0.5);
    }
    if ( strip%128 <= 1 ) {
      adcPrev2 = 0; 
      thePrev2FEDlowThresh  = 9999;
    }
    else if ( strip%128 > 1 ) {
      adcPrev2 = (in_iter-2)->adc();
     thePrev2FEDlowThresh  = static_cast<int16_t>(thresholdHandle->getData(strip-2,detThRange).getLth()*noiseHandle->getNoiseFast(strip-2,detNoiseRange)+0.5);
    }
    //GB 23/6/08: truncation should be done at the very beginning
    if (isAValidDigi())
      out.data.push_back(SiStripDigi(strip, truncate(in_iter->adc())));
  }
}

void SiStripFedZeroSuppression::fillThresholds_(const uint32_t detID, size_t size) {
  SiStripNoises::Range detNoiseRange = noiseHandle->getRange(detID);
  SiStripThreshold::Range detThRange = thresholdHandle->getRange(detID);

  if (highThr_.size() != size) { 
    highThr_.resize(size); 
    lowThr_.resize(size);
    noises_.resize(size);
    highThrSN_.resize(size);
    lowThrSN_.resize(size);
  }
  
  noiseHandle->allNoises(noises_, detNoiseRange);
  thresholdHandle->allThresholds(lowThrSN_, highThrSN_, detThRange); // thresholds as S/N
  for (size_t strip = 0; strip < size; ++strip) {
    float noise     = noises_[strip];
    //  uncomment line below to check bluk noise decoding
    //assert( noise == noiseHandle->getNoiseFast(strip,detNoiseRange) ); 
    highThr_[strip] = static_cast<int16_t>(highThrSN_[strip]*noise+0.5+1e-6);
    lowThr_[strip]  = static_cast<int16_t>( lowThrSN_[strip]*noise+0.5+1e-6);
    // Note: it's a bit wierd, but there are some cases for which 'highThrSN_[strip]*noise' is an exact integer
    //   but due to roundoffs it gets rounded to the integer below if. 
    //   Apparently the optimized code inlines differently and this changes the roundoff.
    //   The +1e-6 fixes the problem.   [GPetruc]
  } 
}


void SiStripFedZeroSuppression::suppress(const std::vector<int16_t>& in, uint16_t firstAPV,  edm::DetSet<SiStripDigi>& out){

  const uint32_t detID = out.id;
  size_t size = in.size();
#ifdef DEBUG_SiStripZeroSuppression_
  if (edm::isDebugEnabled())
    LogTrace("SiStripZeroSuppression") << "[SiStripFedZeroSuppression::suppress] Zero suppression on std::vector<int16_t>: detID " << detID << " size = " << in.size();
#endif

  fillThresholds_(detID, size+firstAPV*128); // want to decouple this from the other cost


  std::vector<int16_t>::const_iterator in_iter=in.begin();
  uint16_t strip = firstAPV*128;
  for (; strip < size+firstAPV*128; ++strip, ++in_iter){

    size_t strip_mod_128 = strip & 127;
#ifdef DEBUG_SiStripZeroSuppression_
    if (edm::isDebugEnabled())
      LogTrace("SiStripZeroSuppression") << "[SiStripFedZeroSuppression::suppress]  detID= " << detID << " strip= " <<  strip << "  adc= " << *in_iter;
#endif    
    adc   = *in_iter;

    theFEDlowThresh  = lowThr_[strip];
    theFEDhighThresh = highThr_[strip];

    //Find adc values for neighbouring strips

     /*
      If a strip is the last one on the chip
      set its next neighbor's thresholds to infinity
      because the FED does not merge clusters across
      chip boundaries right now
    */

    //adcPrev = -9999;  // useless, they are set
    //adcNext = -9999;  // in the next lines in any case
    if ( strip_mod_128 == 127 ) {
      adcNext = 0;      
      theNextFEDlowThresh  = 9999;
      theNextFEDhighThresh = 9999;
    } else {
      adcNext = *(in_iter+1);
      theNextFEDlowThresh  = lowThr_[strip+1];
      theNextFEDhighThresh = highThr_[strip+1];
    }
    
    /*
      Similarily, for the first strip 
      on a chip
    */
    if ( strip_mod_128 == 0 ) {
      adcPrev = 0;
      thePrevFEDlowThresh  = 9999;
      thePrevFEDhighThresh = 9999;   
    } else {
      adcPrev = *(in_iter-1);
      thePrevFEDlowThresh  = lowThr_[strip-1];
      thePrevFEDhighThresh = highThr_[strip-1];
    }

    if ( adcNext < adcPrev){
      adcMaxNeigh = adcPrev;
      theNeighFEDlowThresh  = thePrevFEDlowThresh;
      theNeighFEDhighThresh = thePrevFEDhighThresh;
    } else {
      adcMaxNeigh = adcNext;
      theNeighFEDlowThresh  = theNextFEDlowThresh;
      theNeighFEDhighThresh = theNextFEDhighThresh;
    }
    
    //Find adc values for next neighbouring strips
    //adcPrev2 = -9999;           //
    //adcNext2 = -9999;           // useless to set them here
    //thePrev2FEDlowThresh  = 1;  // they are overwritten always in the next 8 lines
    //theNext2FEDlowThresh  = 1;  //
    if ( strip_mod_128 >=126 ) {
      adcNext2 = 0;
      theNext2FEDlowThresh  = 9999;
    //} else if ( strip_mod_128 < 126 ) { // if it's not >= then is <, no need to "if" again
    } else {
      adcNext2 = *(in_iter+2);
      theNext2FEDlowThresh  = lowThr_[strip+2];
    }
    if ( strip_mod_128 <= 1 ) {
      adcPrev2 = 0; 
      thePrev2FEDlowThresh  = 9999;
    //} else if ( strip_mod_128 > 1 ) { // same as above
    } else {
      adcPrev2 = *(in_iter-2);
      thePrev2FEDlowThresh  = lowThr_[strip-2];;
    }
    
    if (isAValidDigi()){
#ifdef DEBUG_SiStripZeroSuppression_
      if (edm::isDebugEnabled())
	LogTrace("SiStripZeroSuppression") << "[SiStripFedZeroSuppression::suppress] DetId " << out.id << " strip " << strip << " adc " << *in_iter << " digiCollection size " << out.data.size() ;
#endif            
      //GB 23/6/08: truncation should be done at the very beginning
      out.push_back(SiStripDigi(strip, (*in_iter<0 ? 0 : truncate( *in_iter ) )));
    }
  }
}


bool SiStripFedZeroSuppression::isAValidDigi()
{

#ifdef DEBUG_SiStripZeroSuppression_


  if (edm::isDebugEnabled()){

    LogTrace("SiStripZeroSuppression") << "[SiStripFedZeroSuppression::suppress] "
				       << "\n\t adc " << adc 
				       << "\n\t adcPrev " << adcPrev
				       << "\n\t adcNext " << adcNext 
				       << "\n\t adcMaxNeigh " << adcMaxNeigh 
				       << "\n\t adcPrev2 " << adcPrev2 
				       << "\n\t adcNext2 " << adcNext2 
				       <<std::endl;
  
    LogTrace("SiStripZeroSuppression") << "[SiStripFedZeroSuppression::suppress] "
				       << "\n\t theFEDlowThresh " <<  theFEDlowThresh 
				       << "\n\t theFEDhighThresh " << theFEDhighThresh 
				       << "\n\t thePrevFEDlowThresh " <<  thePrevFEDlowThresh 
				       << "\n\t thePrevFEDhighThresh " << thePrevFEDhighThresh 
				       << "\n\t theNextFEDlowThresh " <<  theNextFEDlowThresh 
				       << "\n\t theNextFEDhighThresh " << theNextFEDhighThresh 
				       << "\n\t theNeighFEDlowThresh " <<  theNeighFEDlowThresh 
				       << "\n\t theNeighFEDhighThresh " << theNeighFEDhighThresh 
				       << "\n\t thePrev2FEDlowThresh " <<  thePrev2FEDlowThresh 
				       << "\n\t theNext2FEDlowThresh " <<  theNext2FEDlowThresh
				       <<std::endl;
  }
#endif  
  // Decide if this strip should be accepted.
  bool accept = false;
  switch (theFEDalgorithm) {
  case 1:
    accept = (adc >= theFEDlowThresh);
    break;
  case 2:
    accept = (adc >= theFEDhighThresh || (adc >= theFEDlowThresh &&
					  adcMaxNeigh >= theNeighFEDlowThresh));
    break;
  case 3:
    accept = (adc >= theFEDhighThresh || (adc >= theFEDlowThresh &&
					  adcMaxNeigh >= theNeighFEDhighThresh));
    break;
  case 4:
    accept = (
	      (adc >= theFEDhighThresh)            //Test for adc>highThresh (same as algorithm 2)
	      ||
	      (
	       (adc >= theFEDlowThresh)            //Test for adc>lowThresh, with neighbour adc>lowThresh (same as algorithm 2)
	       &&
	       (adcMaxNeigh >= theNeighFEDlowThresh)
	       ) 
	      ||
	      (
	       (adc < theFEDlowThresh)        //Test for adc<lowThresh
	       &&     
	       (
		(
		 (adcPrev  >= thePrevFEDhighThresh)    //with both neighbours>highThresh
		 &&
		 (adcNext  >= theNextFEDhighThresh)
		 ) 
		||
		(
		 (adcPrev  >= thePrevFEDhighThresh)    //OR with previous neighbour>highThresh and
		 &&
		 (adcNext  >= theNextFEDlowThresh)     //both the next neighbours>lowThresh
		 &&
		 (adcNext2 >= theNext2FEDlowThresh)
		 )  
		||
		(
		 (adcNext  >= theNextFEDhighThresh)    //OR with next neighbour>highThresh and
		 &&
		 (adcPrev  >= thePrevFEDlowThresh)     //both the previous neighbours>lowThresh
		 &&
		 (adcPrev2 >= thePrev2FEDlowThresh)
		 )  
		||
		(
		 (adcNext  >= theNextFEDlowThresh)     //OR with both next neighbours>lowThresh and
		 &&
		 (adcNext2 >= theNext2FEDlowThresh)   //both the previous neighbours>lowThresh
		 &&
		 (adcPrev  >= thePrevFEDlowThresh)  
		 &&
		 (adcPrev2 >= thePrev2FEDlowThresh)
		 )
		)
	       )
	      );
    break;
  case 5:
    accept = adc > 0;
    break;

  }
  return accept;
}


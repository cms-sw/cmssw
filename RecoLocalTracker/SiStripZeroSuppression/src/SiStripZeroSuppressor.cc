#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripZeroSuppressor.h"

using namespace std;

void SiStripZeroSuppressor::suppress(const edm::DetSet<SiStripRawDigi>& in, edm::DetSet<SiStripDigi>& out)
{
  const uint32_t detID = out.id;
  LogDebug("SiStripZeroSuppression") << "[SiStripZeroSuppressor::suppress] Zero suppression on edm::DetSet<SiStripRawDigi>: detID " << detID << " size = " << in.data.size();

  edm::DetSet<SiStripRawDigi>::const_iterator in_iter=in.data.begin();
  for (;in_iter!=in.data.end();in_iter++){

    const uint32_t strip = (uint32_t) (in_iter-in.data.begin());
     
    LogDebug("SiStripZeroSuppression") << "[SiStripZeroSuppressor::suppress] strip = " <<  strip << "  adc = " << in_iter->adc();
    
    adc   = in_iter->adc();
    theFEDlowThresh  = SiStripPedestalsService_->getLowTh(detID,strip)  * SiStripNoiseService_->getNoise(detID,strip);
    theFEDhighThresh = SiStripPedestalsService_->getHighTh(detID,strip) * SiStripNoiseService_->getNoise(detID,strip);

    //Find adc values for neighbouring strips

    //Set values for strips just outside module to infinity.
    //This is to avoid losing strips at the edges, 
    //which otherwise would pass cuts if strips were next to each other. 
    adcPrev = -9999;
    adcNext = -9999;
    if ( in_iter+1 == in.data.end() ) {
      adcNext = 9999;      
      theNextFEDlowThresh  = 1;
      theNextFEDhighThresh = 1;
    }
    else {
      adcNext = (in_iter+1)->adc();
      theNextFEDlowThresh  = SiStripPedestalsService_->getLowTh(detID,strip+1)  * SiStripNoiseService_->getNoise(detID,strip+1);
      theNextFEDhighThresh = SiStripPedestalsService_->getHighTh(detID,strip+1) * SiStripNoiseService_->getNoise(detID,strip+1);
    }
    if ( in_iter == in.data.begin() ) {
      adcPrev = 9999;
      thePrevFEDlowThresh  = 1;
      thePrevFEDhighThresh = 1;   
    }
    else {
      adcPrev = (in_iter-1)->adc();
      thePrevFEDlowThresh  = SiStripPedestalsService_->getLowTh(detID,strip-1)  * SiStripNoiseService_->getNoise(detID,strip-1);
      thePrevFEDhighThresh = SiStripPedestalsService_->getHighTh(detID,strip-1) * SiStripNoiseService_->getNoise(detID,strip-1); 
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
    if ( in_iter  == in.data.end() - 2 ) {
      adcNext2 = 9999;
      theNext2FEDlowThresh  = 1;
    }
    else if ( in_iter < in.data.end() -2 ) {
      adcNext2 = (in_iter+2)->adc();
      theNext2FEDlowThresh  = SiStripPedestalsService_->getLowTh(detID,strip+2) * SiStripNoiseService_->getNoise(detID,strip+2);
    }
    if ( in_iter - in.data.begin() == 1 ) {
      adcPrev2 = 9999; 
      thePrev2FEDlowThresh  = 1;
    }
    else if ( in_iter - in.data.begin() > 1 ) {
      adcPrev2 = (in_iter-2)->adc();
      thePrev2FEDlowThresh  = SiStripPedestalsService_->getLowTh(detID,strip-2) * SiStripNoiseService_->getNoise(detID,strip-2);
    }
    
    if (IsAValidDigi())
      out.data.push_back(SiStripDigi(strip, in_iter->adc()));
  }
}

void SiStripZeroSuppressor::suppress(const std::vector<int16_t>& in, edm::DetSet<SiStripDigi>& out)
{
  const uint32_t detID = out.id;
  LogDebug("SiStripZeroSuppression") << "[SiStripZeroSuppressor::suppress] Zero suppression on std::vector<int16_t>: detID " << detID << " size = " << in.size();

  std::vector<int16_t>::const_iterator in_iter=in.begin();
  for (;in_iter!=in.end();in_iter++){

    const uint32_t strip = (uint32_t) (in_iter-in.begin());
    LogDebug("SiStripZeroSuppression") << "[SiStripZeroSuppressor::suppress]  strip = " <<  strip << "  adc = " << *in_iter;
    
    adc   = *in_iter;
    theFEDlowThresh  = SiStripPedestalsService_->getLowTh(detID,strip)  * SiStripNoiseService_->getNoise(detID,strip);
    theFEDhighThresh = SiStripPedestalsService_->getHighTh(detID,strip) * SiStripNoiseService_->getNoise(detID,strip);

    //Find adc values for neighbouring strips

    //Set values for strips just outside module to infinity.
    //This is to avoid losing strips at the edges, 
    //which otherwise would pass cuts if strips were next to each other. 
    adcPrev = -9999;
    adcNext = -9999;
    if ( in_iter+1 == in.end() ) {
      adcNext = 9999;      
      theNextFEDlowThresh  = 1;
      theNextFEDhighThresh = 1;
    }
    else {
      adcNext = *(in_iter+1);
      theNextFEDlowThresh  = SiStripPedestalsService_->getLowTh(detID,strip+1)  * SiStripNoiseService_->getNoise(detID,strip+1);
      theNextFEDhighThresh = SiStripPedestalsService_->getHighTh(detID,strip+1) * SiStripNoiseService_->getNoise(detID,strip+1);
    }
    if ( in_iter == in.begin() ) {
      adcPrev = 9999;
      thePrevFEDlowThresh  = 1;
      thePrevFEDhighThresh = 1;   
    }
    else {
      adcPrev = *(in_iter-1);
      thePrevFEDlowThresh  = SiStripPedestalsService_->getLowTh(detID,strip-1)  * SiStripNoiseService_->getNoise(detID,strip-1);
      thePrevFEDhighThresh = SiStripPedestalsService_->getHighTh(detID,strip-1) * SiStripNoiseService_->getNoise(detID,strip-1); 
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
    if ( in_iter  == in.end() - 2 ) {
      adcNext2 = 9999;
      theNext2FEDlowThresh  = 1;
    }
    else if ( in_iter < in.end() -2 ) {
      adcNext2 = *(in_iter+2);
      theNext2FEDlowThresh  = SiStripPedestalsService_->getLowTh(detID,strip+2) * SiStripNoiseService_->getNoise(detID,strip+2);
    }
    if ( in_iter - in.begin() == 1 ) {
      adcPrev2 = 9999; 
      thePrev2FEDlowThresh  = 1;
    }
    else if ( in_iter - in.begin() > 1 ) {
      adcPrev2 = *(in_iter-2);
      thePrev2FEDlowThresh  = SiStripPedestalsService_->getLowTh(detID,strip-2) * SiStripNoiseService_->getNoise(detID,strip-2);
    }
    
    if (IsAValidDigi())
      out.data.push_back(SiStripDigi(strip, *in_iter));
  }
}


bool SiStripZeroSuppressor::IsAValidDigi()
{
#ifdef DEBUG
  std::cout << "[SiStripZeroSuppressor::suppress] "
	    << "\n\t adc " << adc 
	    << "\n\t adcPrev " << adcPrev
	    << "\n\t adcNext " << adcNext 
	    << "\n\t adcMaxNeigh " << adcMaxNeigh 
	    << "\n\t adcPrev2 " << adcPrev2 
	    << "\n\t adcNext2 " << adcNext2 
	    <<std::endl;
  
  std::cout << "[SiStripZeroSuppressor::suppress] "
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
  }
  return accept;
}


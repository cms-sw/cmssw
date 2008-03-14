#include <iostream>

#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"  
#include "DQM/SiStripMonitorHardware/interface/Fed9UDebugEvent.hh"
#include "DQM/SiStripMonitorHardware/interface/Fed9UEventAnalyzer.hh"

Fed9UEventAnalyzer::Fed9UEventAnalyzer(std::pair<int,int> newFedBoundaries, bool doSwap) {
  // First of all we instantiate the Fed9U object of the event
  fedEvent_ = new Fed9U::Fed9UDebugEvent(); 
  fedIdBoundaries_ = newFedBoundaries;
  swapOn_=doSwap;
  thisFedId_=0;
}


Fed9UEventAnalyzer::Fed9UEventAnalyzer(Fed9U::u32* data_u32, Fed9U::u32 size_u32,
				       std::pair<int,int> newFedBoundaries,
				       bool doSwap) {
  Fed9UEventAnalyzer(newFedBoundaries,doSwap);

  Initialize(data_u32, size_u32);
}

Fed9UEventAnalyzer::~Fed9UEventAnalyzer() {
  delete fedEvent_;
}

bool Fed9UEventAnalyzer::Initialize(Fed9U::u32* data_u32, Fed9U::u32 size_u32) {

  // Let's perform some cross-check...
  if(data_u32 == NULL){ 
    edm::LogWarning("MissingData") << "Fed9U data pointer is NULL";
    return false;
  }
	
  // Ignores buffers of zero size (container (fed ID) is present but contains nothing) 
  if (!size_u32) {
    edm::LogInfo("MissingData") << "Fed9U data size is zero";
    return false;
  }

  // We are now initializing the fedHeader in order to check that
  // the buffer is SiStripTracker's and it is valid
  FEDHeader fedHeader( reinterpret_cast<const unsigned char*>(data_u32) );

  // Fisrt let's check that the header is not malformed
  if ( ! fedHeader.check() ) {
    edm::LogWarning("CorruptData") << "FED header is corrupt";
    return false;
  }

  // Here we also check that the FEDid corresponds to a tracker one
  thisFedId_=fedHeader.sourceID();
  if ( (thisFedId_<fedIdBoundaries_.first) || (thisFedId_>fedIdBoundaries_.second) ) {
    edm::LogInfo("SkipData") << "FED with ID" << thisFedId_ << "is not a Tracker FED";
    return false;
  }

  // Adjusts the buffer pointers for the DAQ header and trailer present when FRLs are running
  // additonally preforms "flipping" of the bytes in the buffer
  if(swapOn_){

    Fed9U::u32 temp1,temp2;
		
    //32 bit word swapping for the real FED buffers	
    for(unsigned int i = 0; i < (size_u32 - 1); i+=2){	
      temp1 = *(data_u32+i);
      temp2 = *(data_u32+i+1);
      *(data_u32+i) = temp2;
      *(data_u32+i+1) = temp1;
    }
  }
    
  // The actual event initialization, catching its possible exceptions
  try{
    // Initialize the fedEvent with offset for slink
    fedEvent_->Init( data_u32, 0, size_u32 );
  } catch ( const ICUtils::ICException& e ) {
    std::stringstream ss;
    ss << "Caught ICUtils::ICException in fedEvent_->Init with message:"
       << std::endl << e.what();
    edm::LogWarning("FEDEventInitException") << ss.str();
    return false;
  }

  return true;
}


Fed9UErrorCondition Fed9UEventAnalyzer::Analyze() {
  Fed9UErrorCondition result;
  
  // **********************************
  // *                                *
  // * Initialize the result variable *
  // *                                *
  // **********************************

  result.problemsSeen = 0;
  result.totalChannels = fedEvent_->totalChannels();

  // Clear the FED errors
  result.internalFreeze       = false;
  result.bxError              = false;

  // Clear the FPGA errors
  for (unsigned int fpga=0; fpga <8; fpga++) {
    result.feMajorAddress[fpga]  = 0x0;
    result.feOverflow[fpga]      = false;
    result.feEnabled[fpga]       = false;
    result.apvAddressError[fpga] = false;
  }

  // Clear the APV and channel errors
  for (unsigned int channelIndex=0; channelIndex<96; channelIndex++) {
    result.channel[channelIndex]=0;
    result.apv[channelIndex*2]=0;
    result.apv[channelIndex*2+1]=0;
  }

    
  // **********************************
  // *                                *
  // * FED error checks (BE FPGA)     *
  // *                                *
  // **********************************

  // The main error condition is the FED freeze
  if (fedEvent_->getInternalFreeze()) {
    result.internalFreeze = true;
    return result;
  }

  // The second internal condition is the
  // Failure in bx counting (FED screwed)
  if (fedEvent_->getBXError()) {
    result.bxError = true;
    return result;
  }

  result.apveAddress = fedEvent_->getSpecialApvEmulatorAddress();

  // **********************************
  // *                                *
  // * FE FPGA Main cycle and checks  *
  // *                                *
  // **********************************

  // We cycle over the FPGAs of the FED board
  for(unsigned int fpga = 0; fpga < 8 ; fpga++){

    // Enter into the FPGA only if it is enabled
    if( fedEvent_->getFeEnabled(fpga) ){

      result.feEnabled[fpga] = true;
      
      // Look only at FEs with no overflow
      if (fedEvent_->getFeOverflow(fpga)) {

	// Report the overflow
	result.feOverflow[fpga] = true;

      } else {

	// TODO: add APV Address Error bit (i.e. x-check with APVe)

	// FE Address verification portion (for later use)
	result.feMajorAddress[fpga] = fedEvent_->getFeMajorAddress(fpga);
	
	// **********************************
	// *                                *
	// * Checks of 12 fibers in a FE    *
	// *                                *
	// **********************************	
	
	// This cycles over the fibers of the FPGA
	for (unsigned int fi=0; fi<12; fi++) {

	  // Local index to access the result channel vector
	  unsigned channelIndex=12*fpga+fi;


	  bool APV1Error       = fedEvent_->getAPV1Error(fpga,fi);
	  bool APV2Error       = fedEvent_->getAPV2Error(fpga,fi);
	  bool APV1WrongHeader = fedEvent_->getAPV1WrongHeader(fpga,fi);
	  bool APV2WrongHeader = fedEvent_->getAPV2WrongHeader(fpga,fi);
	  bool outOfSync       = fedEvent_->getOutOfSync(fpga,fi);
	  bool unlocked        = fedEvent_->getUnlocked(fpga,fi);

	  bool anyError =
	    APV1Error ||
	    APV2Error ||
	    APV1WrongHeader ||
	    APV2WrongHeader ||
	    outOfSync ||
	    unlocked ;

	  outOfSync     = outOfSync && (!unlocked);

	  bool badAPV1  = (APV1Error || APV1WrongHeader) && (!outOfSync) && (!unlocked);
	  bool badAPV2  = (APV2Error || APV2WrongHeader) && (!outOfSync) && (!unlocked);

	  if (anyError) {
	    result.problemsSeen++;

	    if (unlocked) {
	      result.channel[channelIndex]=FIBERUNLOCKED;
	    }
	    if (outOfSync) {
	      result.channel[channelIndex]=FIBEROUTOFSYNCH;
	    }
	    if (badAPV1) {
	      result.apv[channelIndex*2]=BADAPV;
	    }
	    if (badAPV2) {
	      result.apv[channelIndex*2+1]=BADAPV;
	    }
	  }

	} // Fiber loop end


      } // FE no overflow

    } // if FE enabled
  } // for FE fpga loop 
  
  return result;
}

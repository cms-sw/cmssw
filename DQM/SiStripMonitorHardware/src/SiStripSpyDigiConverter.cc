#include <vector>

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "DataFormats/SiStripCommon/interface/ConstantsForHardwareSystems.h"
#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"

#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"

#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBufferComponents.h"
#include "EventFilter/SiStripRawToDigi/interface/SiStripDetSetVectorFiller.h"

#include "DQM/SiStripMonitorHardware/interface/SiStripFEDSpyBuffer.h"
#include "DQM/SiStripMonitorHardware/interface/SiStripSpyDigiConverter.h"

namespace sistrip {

  std::unique_ptr<SpyDigiConverter::DSVRawDigis> 
  SpyDigiConverter::extractPayloadDigis(const DSVRawDigis* inputScopeDigis,
					std::vector<uint32_t> * pAPVAddresses,
					const bool discardDigisWithAPVAddrErr,
					const sistrip::SpyUtilities::FrameQuality & aQuality,
					const uint16_t expectedPos)
  {
    // Data is already sorted so push back fast into vector to avoid sorts and create DSV later
    std::vector<DetSetRawDigis> outputData;
    outputData.reserve(inputScopeDigis->size());

    //APV address vector indexed by fedid, majority value written.
    pAPVAddresses->resize(sistrip::FED_ID_MAX+1,0);
    std::vector<uint16_t> lAddrVec;
    lAddrVec.reserve(2*sistrip::FEDCH_PER_FED);
    uint16_t lPreviousFedId = 0;
    std::vector<uint16_t> lHeaderBitVec;
    lHeaderBitVec.reserve(sistrip::FEDCH_PER_FED);


    //local DSVRawDigis per FED
    std::vector<DSVRawDigis::const_iterator> lFedScopeDigis;
    lFedScopeDigis.reserve(sistrip::FEDCH_PER_FED);

    // Loop over channels in input collection
    DSVRawDigis::const_iterator inputChannel = inputScopeDigis->begin();
    const DSVRawDigis::const_iterator endChannels = inputScopeDigis->end();
    bool hasBeenProcessed = false;

    for (; inputChannel != endChannels; ++inputChannel) {
            
      // Fill frame parameters. Second parameter is to print debug info (if logDebug enabled....)
      const sistrip::SpyUtilities::Frame lFrame = sistrip::SpyUtilities::extractFrameInfo(*inputChannel,true);

      const uint32_t lFedIndex = inputChannel->detId();
      const uint16_t fedId = static_cast<uint16_t>(lFedIndex/sistrip::FEDCH_PER_FED);
      const uint16_t fedCh = static_cast<uint16_t>(lFedIndex%sistrip::FEDCH_PER_FED);

      if (lPreviousFedId == 0) {
	lPreviousFedId = fedId;
      }

      //print out warning only for non-empty frames....
      if (!sistrip::SpyUtilities::isValid(lFrame,aQuality,expectedPos)){
	//print out only for non-empty frames, else too many prints...
	if (lFrame.firstHeaderBit < sistrip::SPY_SAMPLES_PER_CHANNEL) {
	edm::LogWarning("SiStripSpyDigiConverter") << " FED ID: " << fedId << ", channel: " << fedCh << std::endl
						   << sistrip::SpyUtilities::print(lFrame,
										   std::string("  -- Invalid Frame ")
										   );
	}

	continue;
      }

      //fill local vectors per FED
      if (fedId == lPreviousFedId) {
	if (hasBeenProcessed) hasBeenProcessed = false;
      }
      if (fedId != lPreviousFedId) {
	SpyDigiConverter::processFED(lPreviousFedId,
				     discardDigisWithAPVAddrErr,
				     pAPVAddresses,
				     outputData,
				     lAddrVec,
				     lHeaderBitVec,
				     lFedScopeDigis
				     );
	lPreviousFedId = fedId;
	hasBeenProcessed = true;
      }
      lFedScopeDigis.push_back(inputChannel);
      lAddrVec.push_back(lFrame.apvAddress.first);
      lAddrVec.push_back(lFrame.apvAddress.second);
      lHeaderBitVec.push_back(lFrame.firstHeaderBit);


    } // end of loop over channels.

    //process the last one if not already done.
    if (!hasBeenProcessed) {
      SpyDigiConverter::processFED(lPreviousFedId,
				   discardDigisWithAPVAddrErr,
				   pAPVAddresses,
				   outputData,
				   lAddrVec,
				   lHeaderBitVec,
				   lFedScopeDigis
				   );
    }

    //return DSV of output
    return std::unique_ptr<DSVRawDigis>( new DSVRawDigis(outputData, true) );
    
  } // end of SpyDigiConverter::extractPayloadDigis method


  void SpyDigiConverter::processFED(const uint16_t aPreviousFedId,
				    const bool discardDigisWithAPVAddrErr,
				    std::vector<uint32_t> * pAPVAddresses,
				    std::vector<DetSetRawDigis> & outputData,
				    std::vector<uint16_t> & aAddrVec,
				    std::vector<uint16_t> & aHeaderBitVec,
				    std::vector<DSVRawDigis::const_iterator> & aFedScopeDigis
				    )
  {

    //extract majority address
    uint32_t lMaj = sistrip::SpyUtilities::findMajorityValue(aAddrVec,aPreviousFedId).first;
    if (pAPVAddresses) (*pAPVAddresses)[aPreviousFedId] = lMaj;

    //loop over iterators and fill payload
    std::vector<DSVRawDigis::const_iterator>::iterator lIter;
    unsigned int lCh = 0;
    for (lIter = aFedScopeDigis.begin(); lIter != aFedScopeDigis.end(); ++lIter,++lCh) {

      //discard if APV address different from majority. 
      //Keep if only one of them is wrong: the other APV might be alright ??

      if ( discardDigisWithAPVAddrErr && 
	   aAddrVec[2*lCh] != lMaj && 
	   aAddrVec[2*lCh+1] != lMaj ) {
	continue;
      }

      DetSetRawDigis::const_iterator iDigi = (*lIter)->begin();
      const DetSetRawDigis::const_iterator endOfChannel = (*lIter)->end();

      if (iDigi == endOfChannel) {
	continue;
      }

      //header starts in sample firstHeaderBit and is 18+6 samples long
      const DetSetRawDigis::const_iterator payloadBegin = iDigi+aHeaderBitVec[lCh]+24;
      const DetSetRawDigis::const_iterator payloadEnd = payloadBegin + STRIPS_PER_FEDCH;
              
      if(payloadEnd-iDigi >= endOfChannel-iDigi) continue; // few-cases where this is possible, i.e. nothing above frame-threhsold                                                                

      // Copy data into output collection
      // Create new detSet with same key (in this case it is the fedKey, not detId)
      outputData.push_back( DetSetRawDigis((*lIter)->detId()) );
      std::vector<SiStripRawDigi>& outputDetSetData = outputData.back().data;
      outputDetSetData.resize(STRIPS_PER_FEDCH);
      std::vector<SiStripRawDigi>::iterator outputBegin = outputDetSetData.begin();
      std::copy(payloadBegin, payloadEnd, outputBegin);

    }

    aFedScopeDigis.clear();
    aAddrVec.clear();
    aHeaderBitVec.clear();

    aAddrVec.reserve(2*sistrip::FEDCH_PER_FED);
    aHeaderBitVec.reserve(sistrip::FEDCH_PER_FED);
    aFedScopeDigis.reserve(sistrip::FEDCH_PER_FED);


  }




  std::unique_ptr<SpyDigiConverter::DSVRawDigis> SpyDigiConverter::reorderDigis(const DSVRawDigis* inputPayloadDigis)
  {
    // Data is already sorted so push back fast into vector to avoid sorts and create DSV later
    std::vector<DetSetRawDigis> outputData;
    outputData.reserve(inputPayloadDigis->size());
    
    // Loop over channels in input collection
    for (DSVRawDigis::const_iterator inputChannel = inputPayloadDigis->begin(); inputChannel != inputPayloadDigis->end(); ++inputChannel) {
      const std::vector<SiStripRawDigi>& inputDetSetData = inputChannel->data;
      // Create new detSet with same key (in this case it is the fedKey, not detId)
      outputData.push_back( DetSetRawDigis(inputChannel->detId()) );
      std::vector<SiStripRawDigi>& outputDetSetData = outputData.back().data;
      outputDetSetData.resize(STRIPS_PER_FEDCH);
      // Copy the data into the output vector reordering
      for (uint16_t readoutOrderStripIndex = 0; readoutOrderStripIndex < inputDetSetData.size(); ++readoutOrderStripIndex) {
	const uint16_t physicalOrderStripIndex = FEDStripOrdering::physicalOrderForStripInChannel(readoutOrderStripIndex);
	outputDetSetData.at(physicalOrderStripIndex) = inputDetSetData.at(readoutOrderStripIndex);
      }
    }
    
    //return DSV of output
    return std::unique_ptr<DSVRawDigis>( new DSVRawDigis(outputData,true) );
  } // end of SpyDigiConverter::reorderDigis method.

  std::unique_ptr<SpyDigiConverter::DSVRawDigis>
  SpyDigiConverter::mergeModuleChannels(const DSVRawDigis* inputPhysicalOrderChannelDigis, 
					const SiStripFedCabling& cabling)
  {
    // Create filler for detSetVector to create output (with maximum number of DetSets and digis)
    uint16_t nFeds = static_cast<uint16_t>( FED_ID_MAX - FED_ID_MIN + 1);

    RawDigiDetSetVectorFiller dsvFiller(nFeds*FEDCH_PER_FED/2, nFeds*FEDCH_PER_FED*STRIPS_PER_FEDCH);
    // Loop over FEDs in cabling
    auto iFed = cabling.fedIds().begin();
    auto endFeds = cabling.fedIds().end();
    for (; iFed != endFeds; ++iFed) {
      // Loop over cabled channels
      auto conns = cabling.fedConnections(*iFed);
      auto iConn = conns.begin();
      auto endConns = conns.end();
      for (; iConn != endConns; ++iConn) {
	// Skip channels not connected to a detector.
	if (!iConn->isConnected()) continue;
	if (iConn->detId() == sistrip::invalid32_) continue;
                
	// Find the data from the input collection
	const uint32_t fedIndex = ( ( iConn->detId()  & sistrip::invalid_ ) << 16 ) | ( iConn->fedCh() & sistrip::invalid_ ) ;
	const DSVRawDigis::const_iterator iDetSet = inputPhysicalOrderChannelDigis->find(fedIndex);
	if (iDetSet == inputPhysicalOrderChannelDigis->end()) {
	  // NOTE: It will display this warning if channel hasn't been unpacked...
	  // Will comment out for now.
	  //edm::LogWarning("SiStripSpyDigiConverter") << "No data found for FED ID: " << iConn->fedId() << " channel: " << iConn->fedCh();
	  continue;
	}
                
	// Start a new channel indexed by the detId in the filler
	dsvFiller.newChannel(iConn->detId(),iConn->apvPairNumber()*STRIPS_PER_FEDCH);
                
	// Add the data
	DetSetRawDigis::const_iterator iDigi = iDetSet->begin();
	const DetSetRawDigis::const_iterator endDetSetDigis = iDetSet->end();
	for (; iDigi != endDetSetDigis; ++iDigi) {
	  dsvFiller.addItem(*iDigi);
	} // end of loop over the digis.
      } // end of loop over channels.
    } // end of loop over FEDs
        
    return dsvFiller.createDetSetVector();
  } // end of SpyDigiConverter::mergeModuleChannels method.



} // end of sistrip namespace.

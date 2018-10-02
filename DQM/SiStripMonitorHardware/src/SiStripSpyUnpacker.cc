/* \file SiStripSpyUnpacker.cc
 * \brief Source code for the SpyChannel unpacking factory.
 */
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSet.h"

#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"

#include "EventFilter/SiStripRawToDigi/interface/SiStripDetSetVectorFiller.h"

#include "DQM/SiStripMonitorHardware/interface/SiStripFEDSpyBuffer.h"
#include "DQM/SiStripMonitorHardware/interface/SiStripSpyUnpacker.h"

#include <algorithm>
#include <vector>
#include <ext/algorithm>

using namespace std;

namespace sistrip {

  SpyUnpacker::SpyUnpacker(const bool allowIncompleteEvents) :
    allowIncompleteEvents_(allowIncompleteEvents)
  {
    if ( edm::isDebugEnabled() ) {
      LogTrace("SiStripSpyUnpacker")
	<< "[sistrip::SpyUnpacker::"<<__func__<<"]"
	<<" Constructing object...";
    }
  } // end of SpyUnpacker constructor.

  SpyUnpacker::~SpyUnpacker() {
    if ( edm::isDebugEnabled() ) {
      LogTrace("SiStripSpyUnpacker")
	<< "[sistrip::SpyUnpacker::"<<__func__<<"]"
	<< " Destructing object...";
    }
  } // end of SpyUnpacker destructor.

  void SpyUnpacker::createDigis(const SiStripFedCabling& cabling,
				const FEDRawDataCollection& buffers,
				RawDigis* pDigis,
				const std::vector<uint32_t> & ids,
				Counters* pTotalEventCounts,
				Counters* pL1ACounts,
				uint32_t* aRunRef)
  {
    //create DSV filler to fill output

    //As of Feb 2010, bug in DataFormats/SiStripCommon/interface/ConstantsForHardwareSystems.h: 
    //number of feds=max-min+1 (440...). Corrected in head of DataFormats/SiStripCommon package.

    uint16_t nFeds = static_cast<uint16_t>( FED_ID_MAX - FED_ID_MIN + 1);
    
    RawDigiDetSetVectorFiller dsvFiller(nFeds*FEDCH_PER_FED,nFeds*FEDCH_PER_FED*SPY_SAMPLES_PER_CHANNEL);
        
    //check if FEDs found in cabling map and event data
    if ( edm::isDebugEnabled() ) {
      if ( cabling.fedIds().empty() ) {
	edm::LogWarning("SiStripSpyUnpacker")
	  << "[sistrip::SpyUnpacker::" << __func__ << "]"
	  << " No FEDs found in cabling map!";
      }
    }
 
    //retrieve FED ids from cabling map and iterate through 
    std::vector<uint32_t>::const_iterator     ifed = ids.begin();
    std::vector<uint32_t>::const_iterator   endfed = ids.end();

    //reference value for run number
    //encoded per fed but should be the same for all feds
    uint32_t lRef = 0;

    //initialise counter vectors to FED_ID_MAX+1
    pTotalEventCounts->resize(FED_ID_MAX+1,0);
    pL1ACounts->resize(FED_ID_MAX+1,0);
    
    for ( ; ifed != endfed; ++ifed ) {

      uint32_t lFedId = (*ifed);

      //check the fedid is valid:
      if (lFedId < FED_ID_MIN || lFedId > FED_ID_MAX) {
	if ( edm::isDebugEnabled() ) {
	  edm::LogWarning("SiStripSpyUnpacker")
	    << "[sistrip::SpyUnpacker::" << __func__ << "]"
	    << " Invalid FED id provided: " 
	    << lFedId;
	}
	continue;
      }

      //retrieve FED raw data for given FED 
      const FEDRawData& input = buffers.FEDData( static_cast<int>(lFedId) );
      //check on FEDRawData pointer
      if ( !input.data() ) {
	if ( edm::isDebugEnabled() ) {
	  edm::LogWarning("SiStripSpyUnpacker")
	    << "[sistrip::SpyUnpacker::" << __func__ << "]"
	    << " NULL pointer to FEDRawData for FED id " 
	    << lFedId;
	}
	continue;
      }
      //check on FEDRawData size
      if ( !input.size() ) {
	if ( edm::isDebugEnabled() ) {
	  edm::LogWarning("SiStripSpyUnpacker")
	    << "[sistrip::SpyUnpacker::" << __func__ << "]"
	    << " FEDRawData has zero size for FED id " 
	    << lFedId;
	}
	continue;
      }
          
      //get the cabling connections for this FED
      auto conns = cabling.fedConnections(lFedId);
          
      //construct FEDBuffer
      std::unique_ptr<sistrip::FEDSpyBuffer> buffer;
      try {
	buffer.reset(new sistrip::FEDSpyBuffer(input.data(),input.size()));
	if (!buffer->doChecks() && !allowIncompleteEvents_) {
	  throw cms::Exception("FEDSpyBuffer") << "FED Buffer check fails for FED ID " << lFedId << ".";
	}
      } catch (const cms::Exception& e) { 
	if ( edm::isDebugEnabled() ) {
	  edm::LogWarning("SiStripSpyUnpacker")
	    << "Exception caught when creating FEDSpyBuffer object for FED " << lFedId << ": " << e.what();
	}
	continue;
      } // end of buffer reset try.
        
      // Get the event counter values
      uint32_t totalEvCount = buffer->spyHeaderTotalEventCount();
      uint32_t l1ID         = buffer->spyHeaderL1ID(); 

      uint32_t lGRun = buffer->globalRunNumber();

      //for the first fed, put it as reference value for others
      if (lRef == 0) lRef = lGRun;
      if (lGRun != lRef){
	edm::LogError("SiStripSpyUnpacker")
	  << " -- Global run encoded in buffer for FED " << lFedId << ": " 
	  << lGRun << " is different from reference value " << lRef 
	  << std::endl;
      }

      // Add event counters
      (*pL1ACounts)[lFedId] = l1ID;
      (*pTotalEventCounts)[lFedId] = totalEvCount;

      //iterate through FED channels, extract payload and create Digis
      std::vector<FedChannelConnection>::const_iterator iconn = conns.begin();
      std::vector<FedChannelConnection>::const_iterator endconn = conns.end();
      for ( ; iconn != endconn; ++iconn ) {

	//check if fed connection is valid
	if ( !iconn->isConnected()) { continue; }

	//FED channel
	uint16_t chan = iconn->fedCh();

	//check values are valid:
	if (chan > FEDCH_PER_FED || iconn->fedId() != lFedId){
	  if (edm::isDebugEnabled()) {
	    std::ostringstream ss;
	    ss << "Channel connection values invalid: iconn->fedId() = " << iconn->fedId() << " for FED " << lFedId << ", iconn->fedCh() = " << chan << std::endl;
	    edm::LogWarning("SiStripSpyUnpacker") << ss.str();
	  }
	  continue;
	}

	//check FED channel
	if (!buffer->channelGood(chan)) {
	  if (edm::isDebugEnabled()) {
	    std::ostringstream ss;
	    ss << "Channel check failed for FED " << lFedId << " channel " << chan << std::endl;
	    edm::LogWarning("SiStripSpyUnpacker") << ss.str();
	  }
	  continue;
	}

	//determine key from cabling	
	const uint32_t key = ( ( lFedId & sistrip::invalid_ ) << 16 ) | ( chan & sistrip::invalid_ );

	// Start a new channel in the filler
	dsvFiller.newChannel(key);
	// Create the unpacker object
	sistrip::FEDSpyChannelUnpacker unpacker = sistrip::FEDSpyChannelUnpacker(buffer->channel(chan));

	// Unpack the data into dsv filler
	while (unpacker.hasData()) {
	  dsvFiller.addItem(unpacker.adc());
	  unpacker++;
	}
      } // end of channel loop

    } //fed loop
    
    //set the run number
    *aRunRef = lRef;

    //create DSV to return
    std::unique_ptr<RawDigis> pResult = dsvFiller.createDetSetVector();
    pDigis->swap(*pResult);

  } // end of SpyUnpacker::createDigis method.

} // end of sistrip namespace.

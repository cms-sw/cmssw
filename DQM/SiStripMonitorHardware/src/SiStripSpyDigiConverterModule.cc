#include <string>
#include <memory>
#include <vector>
#include <utility>
#include "boost/cstdint.hpp"

#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"

#include "DQM/SiStripMonitorHardware/interface/SiStripSpyDigiConverter.h"
#include "DQM/SiStripMonitorHardware/interface/SiStripSpyUtilities.h"

class SiStripFedCabling;

/**
   @file DQM/SiStripMonitorHardware/interface/SiStripSpyDigiConverterModule.cc
   @class SiStripSpyDigiConverterModule 
   
   @brief A plug-in module that takes the spy channel scope mode digis as input
   from the Event and creates EDProducts containing intermediate and VirginRaw
   like SiStripRawDigis. 
*/

namespace sistrip {

  class SpyDigiConverterModule : public edm::EDProducer
  {
  public:
    SpyDigiConverterModule( const edm::ParameterSet& );
    ~SpyDigiConverterModule() override;
    void produce( edm::Event&, const edm::EventSetup& ) override;

  private:
    const edm::InputTag productLabel_;
    edm::EDGetTokenT<sistrip::SpyDigiConverter::DSVRawDigis> productToken_;
    const bool storeAPVAddress_;             //!< True = store APVE address for each channel.
    const bool storePayloadDigis_, storeReorderedDigis_, storeModuleDigis_;
    sistrip::SpyUtilities::FrameQuality frameQuality_;
    const bool discardDigisWithAPVAddressError_;
    const uint32_t expectedHeaderBit_;
    
    //utilities for cabling etc...
    SpyUtilities utility_;
  };

} // end of sistrip namespace.



namespace sistrip {

  SpyDigiConverterModule::SpyDigiConverterModule( const edm::ParameterSet& pset ) :
    productLabel_(pset.getParameter<edm::InputTag>("InputProductLabel")),
    storeAPVAddress_(pset.getParameter<bool>("StoreAPVAddress")),
    storePayloadDigis_(pset.getParameter<bool>("StorePayloadDigis")),
    storeReorderedDigis_(pset.getParameter<bool>("StoreReorderedDigis")),
    storeModuleDigis_(pset.getParameter<bool>("StoreModuleDigis")),
    discardDigisWithAPVAddressError_(pset.getParameter<bool>("DiscardDigisWithWrongAPVAddress")),
    expectedHeaderBit_(pset.getParameter<uint32_t>("ExpectedPositionOfFirstHeaderBit"))
  {
    productToken_ = consumes<sistrip::SpyDigiConverter::DSVRawDigis>(productLabel_);
    if ( edm::isDebugEnabled() ) {
      LogTrace("SiStripSpyDigiConverter")
	<< "[sistrip::SpyDigiConverterModule::" << __func__ << "]"
	<< " Constructing object...";
    }
    
    if (storePayloadDigis_)   produces< edm::DetSetVector<SiStripRawDigi> >("Payload");
    if (storeReorderedDigis_) produces< edm::DetSetVector<SiStripRawDigi> >("Reordered");
    if (storeModuleDigis_)    produces< edm::DetSetVector<SiStripRawDigi> >("VirginRaw");
        
    if (storeAPVAddress_) {
      produces< std::vector<uint32_t> >("APVAddress");
      //produces<uint8_t>("APVAddress");
    }


    frameQuality_.minDigiRange = static_cast<uint16_t>(pset.getParameter<uint32_t>("MinDigiRange"));
    frameQuality_.maxDigiRange = static_cast<uint16_t>(pset.getParameter<uint32_t>("MaxDigiRange"));
    frameQuality_.minZeroLight = static_cast<uint16_t>(pset.getParameter<uint32_t>("MinZeroLight"));
    frameQuality_.maxZeroLight = static_cast<uint16_t>(pset.getParameter<uint32_t>("MaxZeroLight"));
    frameQuality_.minTickHeight = static_cast<uint16_t>(pset.getParameter<uint32_t>("MinTickHeight"));
    frameQuality_.maxTickHeight = static_cast<uint16_t>(pset.getParameter<uint32_t>("MaxTickHeight"));

    
  } // end of constructor.


  SpyDigiConverterModule::~SpyDigiConverterModule() {
    if ( edm::isDebugEnabled() ) {
      LogTrace("SiStripSpyDigiConverter")
	<< "[sistrip::SpyDigiConverterModule::" << __func__ << "]"
	<< " Destructing object...";
    }
  } // end of destructor.

  /** 
      Retrieves cabling map from EventSetup and spy channel scope mode digis
      from Event, creates DetSetVectors of SiStripRawDigis, at verious levels of
      processing, using the SiStripSpyDigiConverter class and
      attaches the containers to the Event.
  */
  void SpyDigiConverterModule::produce( edm::Event& event, const edm::EventSetup& setup ) {

    static bool lFirstEvent = true;

    const SiStripFedCabling* lCabling = utility_.getCabling( setup );

    if ( !(storePayloadDigis_ || storeReorderedDigis_ || storeModuleDigis_ || storeAPVAddress_) ) return;
    
    //retrieve the scope mode digis
    edm::Handle<sistrip::SpyDigiConverter::DSVRawDigis> scopeDigisHandle;
    event.getByToken(productToken_, scopeDigisHandle);
    
    //32-bit to accomodate known CMSSW container
    std::unique_ptr< std::vector<uint32_t> > pAPVAddresses(new std::vector<uint32_t>);
        
    std::unique_ptr<sistrip::SpyDigiConverter::DSVRawDigis> payloadDigis, reorderedDigis, moduleDigis;
    
    //get the majority value for expected position of first header bit
    //from first event, compare to expected one, else output warning.
    if (lFirstEvent){
      uint16_t lFirstHeaderBit;
      sistrip::SpyUtilities::getMajorityHeader(scopeDigisHandle.product(),lFirstHeaderBit);

      if (lFirstHeaderBit != static_cast<uint16_t>(expectedHeaderBit_)) {
	edm::LogWarning("") << " -- Majority position for firstHeaderBit in first event ("
			    << lFirstHeaderBit
			    << ") is not where expected: "
			    << static_cast<uint16_t>(expectedHeaderBit_)
			    << std::endl;
      }

    }



    //extract frame digis and APV addresses
    payloadDigis = sistrip::SpyDigiConverter::extractPayloadDigis(scopeDigisHandle.product(), 
								  pAPVAddresses.get(),
								  discardDigisWithAPVAddressError_,
								  frameQuality_, 
								  static_cast<uint16_t>(expectedHeaderBit_));
        
    // Reorder
    if (storeReorderedDigis_ || storeModuleDigis_) {
      reorderedDigis = sistrip::SpyDigiConverter::reorderDigis(payloadDigis.get());
    }
    
    // Merge into modules
    if (storeModuleDigis_) {
      moduleDigis = sistrip::SpyDigiConverter::mergeModuleChannels(reorderedDigis.get(), *lCabling);
    }
    
    //add to event
    if (storePayloadDigis_) event.put(std::move(payloadDigis),"Payload");
    if (storeReorderedDigis_) event.put(std::move(reorderedDigis),"Reordered");
    if (storeModuleDigis_) event.put(std::move(moduleDigis),"VirginRaw");
    if (storeAPVAddress_) {
      event.put(std::move(pAPVAddresses), "APVAddress");
    }
        

    lFirstEvent = false;

  } // end of SpyDigiConverter::produce method.

} // end of sistrip namespace.


#include "FWCore/Framework/interface/MakerMacros.h"
typedef sistrip::SpyDigiConverterModule SiStripSpyDigiConverterModule;
DEFINE_FWK_MODULE(SiStripSpyDigiConverterModule);

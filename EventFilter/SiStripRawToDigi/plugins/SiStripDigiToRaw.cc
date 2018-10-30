
#include "SiStripDigiToRaw.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "FWCore/Utilities/interface/CRC16.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBuffer.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <boost/format.hpp>

namespace sistrip {

  // -----------------------------------------------------------------------------
  /** */
  DigiToRaw::DigiToRaw( FEDReadoutMode mode,
                        uint8_t packetCode,
                        bool useFedKey ) :
    mode_(mode),
    packetCode_(packetCode),
    useFedKey_(useFedKey),
    bufferGenerator_(),
    warnings_("DigiToRaw", "[sistrip::DigiToRaw::createFedBuffers_]", edm::isDebugEnabled())
  {
    if ( edm::isDebugEnabled() ) {
      LogDebug("DigiToRaw")
        << "[sistrip::DigiToRaw::DigiToRaw]"
        << " Constructing object...";
    }
    bufferGenerator_.setReadoutMode(mode_);
  }

  // -----------------------------------------------------------------------------
  /** */
  DigiToRaw::~DigiToRaw() {
    if ( edm::isDebugEnabled() ) {
      LogDebug("DigiToRaw")
        << "[sistrip::DigiToRaw::~DigiToRaw]"
        << " Destructing object...";
    }
  }

  // -----------------------------------------------------------------------------
  /** 
      Input: DetSetVector of SiStripDigis. Output: FEDRawDataCollection.
      Retrieves and iterates through FED buffers, extract FEDRawData
      from collection and (optionally) dumps raw data to stdout, locates
      start of FED buffer by identifying DAQ header, creates new
      Fed9UEvent object using current FEDRawData buffer, dumps FED
      buffer to stdout, retrieves data from various header fields
  */
  void DigiToRaw::createFedBuffers( edm::Event& event,
                                    edm::ESHandle<SiStripFedCabling>& cabling,
                                    edm::Handle< edm::DetSetVector<SiStripDigi> >& collection,
                                    std::unique_ptr<FEDRawDataCollection>& buffers ) {
    createFedBuffers_(event, cabling, collection, buffers, true);
  }
  
  void DigiToRaw::createFedBuffers( edm::Event& event,
                                    edm::ESHandle<SiStripFedCabling>& cabling,
                                    edm::Handle< edm::DetSetVector<SiStripRawDigi> >& collection,
                                    std::unique_ptr<FEDRawDataCollection>& buffers ) {
    createFedBuffers_(event, cabling, collection, buffers, false);
  }

  //with copy of headers from initial raw data collection (raw->digi->raw)
  void DigiToRaw::createFedBuffers( edm::Event& event,
                                    edm::ESHandle<SiStripFedCabling>& cabling,
                                    edm::Handle<FEDRawDataCollection> & rawbuffers,
                                    edm::Handle< edm::DetSetVector<SiStripDigi> >& collection,
                                    std::unique_ptr<FEDRawDataCollection>& buffers ) {
    createFedBuffers_(event, cabling, rawbuffers, collection, buffers, true);
  }
  
  void DigiToRaw::createFedBuffers( edm::Event& event,
                                    edm::ESHandle<SiStripFedCabling>& cabling,
                                    edm::Handle<FEDRawDataCollection> & rawbuffers,
                                    edm::Handle< edm::DetSetVector<SiStripRawDigi> >& collection,
                                    std::unique_ptr<FEDRawDataCollection>& buffers ) {
    createFedBuffers_(event, cabling, rawbuffers, collection, buffers, false);
  }




  template<class Digi_t>
  void DigiToRaw::createFedBuffers_( edm::Event& event,
                                     edm::ESHandle<SiStripFedCabling>& cabling,
                                     edm::Handle< edm::DetSetVector<Digi_t> >& collection,
                                     std::unique_ptr<FEDRawDataCollection>& buffers,
                                     bool zeroSuppressed) {

    edm::Handle<FEDRawDataCollection> rawbuffers;
    //CAMM initialise to some dummy empty buffers??? Or OK like this ?
    createFedBuffers_(event,cabling,rawbuffers,collection,buffers,zeroSuppressed);

  }

  template<class Digi_t>
  void DigiToRaw::createFedBuffers_( edm::Event& event,
                                     edm::ESHandle<SiStripFedCabling>& cabling,
                                     edm::Handle<FEDRawDataCollection> & rawbuffers,
                                     edm::Handle< edm::DetSetVector<Digi_t> >& collection,
                                     std::unique_ptr<FEDRawDataCollection>& buffers,
                                     bool zeroSuppressed) {
    const bool dataIsAlready8BitTruncated = zeroSuppressed && ( ! (
          //for special mode premix raw, data is zero-suppressed but not converted to 8 bit
             ( mode_ == READOUT_MODE_PREMIX_RAW )
          // the same goes for 10bit ZS modes
          || ( ( mode_ == READOUT_MODE_ZERO_SUPPRESSED ) && ( packetCode_ == PACKET_CODE_ZERO_SUPPRESSED10 ) )
          || ( mode_ == READOUT_MODE_ZERO_SUPPRESSED_LITE10 )
          || ( mode_ == READOUT_MODE_ZERO_SUPPRESSED_LITE10_CMOVERRIDE )
          ) );
    try {
      
      //set the L1ID to use in the buffers
      bufferGenerator_.setL1ID(0xFFFFFF & event.id().event());
      auto fed_ids = cabling->fedIds();

      //copy header if valid rawbuffers handle
      if (rawbuffers.isValid()){
        if ( edm::isDebugEnabled() ) {
          edm::LogWarning("DigiToRaw")
            << "[sistrip::DigiToRaw::createFedBuffers_]"
            << " Valid raw buffers, getting headers from them..."
            << " Number of feds: " << fed_ids.size() << " between "
            << *(fed_ids.begin()) << " and " << *(fed_ids.end());
        }

        const FEDRawDataCollection& rawDataCollection = *rawbuffers;

        for ( auto ifed = fed_ids.begin(); ifed != fed_ids.end(); ++ifed ) {
          const FEDRawData& rawfedData = rawDataCollection.FEDData(*ifed);

          if ( edm::isDebugEnabled() ) {
            edm::LogWarning("DigiToRaw")
              << "[sistrip::DigiToRaw::createFedBuffers_]"
              << "Fed " << *ifed << " : size of buffer = " << rawfedData.size();
          }

          //need to construct full object to copy full header
          std::unique_ptr<const sistrip::FEDBuffer> fedbuffer;
          if ( rawfedData.size() == 0 ) {
            warnings_.add("Invalid raw data for FED, skipping", (boost::format("id %1%") % *ifed).str());
          } else {
            try {
              fedbuffer.reset(new sistrip::FEDBuffer(rawfedData.data(),rawfedData.size(),true));
            } catch (const cms::Exception& e) {
              edm::LogWarning("DigiToRaw") << "[sistrip::DigiToRaw::createFedBuffers_]"
                        << " Could not construct FEDBuffer for FED " << *ifed
                        << std::endl;
            }

            if ( edm::isDebugEnabled() ) {
              edm::LogWarning("DigiToRaw")
                << "[sistrip::DigiToRaw::createFedBuffers_]"
                << " Original header type: " << fedbuffer->headerType();
            }

            bufferGenerator_.setHeaderType(FEDHeaderType::HEADER_TYPE_FULL_DEBUG);
            //fedbuffer->headerType());
            bufferGenerator_.daqHeader() = fedbuffer->daqHeader();
            bufferGenerator_.daqTrailer() = fedbuffer->daqTrailer();

            bufferGenerator_.trackerSpecialHeader() = fedbuffer->trackerSpecialHeader();
            bufferGenerator_.setReadoutMode(mode_);

            if ( edm::isDebugEnabled() ) {
              std::ostringstream debugStream;
              if(ifed==fed_ids.begin()){  bufferGenerator_.trackerSpecialHeader().print(std::cout); std::cout << std::endl;}
              bufferGenerator_.trackerSpecialHeader().print(debugStream);
              edm::LogWarning("DigiToRaw")
                << "[sistrip::DigiToRaw::createFedBuffers_]"
                << " Tracker special header apveAddress: " << static_cast<int>(bufferGenerator_.trackerSpecialHeader().apveAddress())
                << " - event type = " << bufferGenerator_.getDAQEventType()
                << " - buffer format = " << bufferGenerator_.getBufferFormat() << "\n"
                << " - SpecialHeader bufferformat " << bufferGenerator_.trackerSpecialHeader().bufferFormat()
                << " - headertype " << bufferGenerator_.trackerSpecialHeader().headerType()
                << " - readoutmode " << bufferGenerator_.trackerSpecialHeader().readoutMode()
                << " - apvaddrregister " << std::hex << static_cast<int>(bufferGenerator_.trackerSpecialHeader().apvAddressErrorRegister())
                << " - feenabledregister " << std::hex << static_cast<int>(bufferGenerator_.trackerSpecialHeader().feEnableRegister())
                << " - feoverflowregister " << std::hex << static_cast<int>(bufferGenerator_.trackerSpecialHeader().feOverflowRegister())
                << " - statusregister " << bufferGenerator_.trackerSpecialHeader().fedStatusRegister() << "\n"
                << " SpecialHeader: " << debugStream.str();
            }

            std::unique_ptr<FEDFEHeader> tempFEHeader(fedbuffer->feHeader()->clone());
            FEDFullDebugHeader* fedFeHeader = dynamic_cast<FEDFullDebugHeader*>(tempFEHeader.get());
            if ( edm::isDebugEnabled() ) {
              std::ostringstream debugStream;
              if(ifed==fed_ids.begin()){  std::cout << "FEHeader before transfer: " << std::endl;fedFeHeader->print(std::cout);std::cout  <<std::endl;}
              fedFeHeader->print(debugStream);
              edm::LogWarning("DigiToRaw")
                << "[sistrip::DigiToRaw::createFedBuffers_]"
                << " length of original feHeader: " << fedFeHeader->lengthInBytes() << "\n"
                << debugStream.str();
            }
            //status registers
            (bufferGenerator_.feHeader()).setBEStatusRegister(fedFeHeader->beStatusRegister());
            (bufferGenerator_.feHeader()).setDAQRegister2(fedFeHeader->daqRegister2());
            (bufferGenerator_.feHeader()).setDAQRegister(fedFeHeader->daqRegister());
            for(uint8_t iFE=1; iFE<6; iFE++) {
              (bufferGenerator_.feHeader()).set32BitReservedRegister(iFE,fedFeHeader->get32BitWordFrom(fedFeHeader->feWord(iFE)+10));
            }

            std::vector<bool> feEnabledVec;
            feEnabledVec.resize(FEUNITS_PER_FED,true);
            for (uint8_t iFE = 0; iFE < FEUNITS_PER_FED; iFE++) {
              feEnabledVec[iFE]=fedbuffer->trackerSpecialHeader().feEnabled(iFE);  
              (bufferGenerator_.feHeader()).setFEUnitMajorityAddress(iFE,fedFeHeader->feUnitMajorityAddress(iFE));
              for (uint8_t iFEUnitChannel = 0; iFEUnitChannel < FEDCH_PER_FEUNIT; iFEUnitChannel++) {
                (bufferGenerator_.feHeader()).setChannelStatus(iFE,iFEUnitChannel,fedFeHeader->getChannelStatus(iFE,iFEUnitChannel));
              }//loop on channels
            }//loop on fe units
            bufferGenerator_.setFEUnitEnables(feEnabledVec);

            if ( edm::isDebugEnabled() ) {
              std::ostringstream debugStream;
              if(ifed==fed_ids.begin()){  std::cout << "\nFEHeader after transfer: " << std::endl;bufferGenerator_.feHeader().print(std::cout); std::cout << std::endl;}
              bufferGenerator_.feHeader().print(debugStream);
              edm::LogWarning("DigiToRaw")
                << "[sistrip::DigiToRaw::createFedBuffers_]"
                << " length of feHeader: " << bufferGenerator_.feHeader().lengthInBytes() << "\n"
                << debugStream.str();
            }
            auto conns = cabling->fedConnections(*ifed);
            
            FEDStripData fedData(dataIsAlready8BitTruncated);
            
            
            for (auto iconn = conns.begin() ; iconn != conns.end(); iconn++ ) {
              
              // Determine FED key from cabling
              uint32_t fed_key = ( ( iconn->fedId() & sistrip::invalid_ ) << 16 ) | ( iconn->fedCh() & sistrip::invalid_ );
              
              // Determine whether DetId or FED key should be used to index digi containers
              uint32_t key = ( useFedKey_ || mode_ == READOUT_MODE_SCOPE ) ? fed_key : iconn->detId();
              
              // Check key is non-zero and valid
              if ( !key || ( key == sistrip::invalid32_ ) ) { continue; }
              
              // Determine APV pair number (needed only when using DetId)
              uint16_t ipair = ( useFedKey_ || mode_ == READOUT_MODE_SCOPE ) ? 0 : iconn->apvPairNumber();
              
              FEDStripData::ChannelData& chanData = fedData[iconn->fedCh()];

              // Find digis for DetID in collection
              if (!collection.isValid()){
                if ( edm::isDebugEnabled() ) {
                  edm::LogWarning("DigiToRaw") 
                  << "[DigiToRaw::createFedBuffers] " 
                  << "digis collection is not valid...";
                }
                break;
              }
              typename std::vector< edm::DetSet<Digi_t> >::const_iterator digis = collection->find( key );
              if (digis == collection->end()) { continue; } 

              typename edm::DetSet<Digi_t>::const_iterator idigi, digis_begin(digis->data.begin());
              for ( idigi = digis_begin; idigi != digis->data.end(); idigi++ ) {
                
                if ( STRIP(idigi, digis_begin) < ipair*256 ||
                   STRIP(idigi, digis_begin) > ipair*256+255 ) { continue; }
                const unsigned short strip = STRIP(idigi, digis_begin) % 256;
                
                if ( strip >= STRIPS_PER_FEDCH ) {
                  if ( edm::isDebugEnabled() ) {
                  std::stringstream ss;
                  ss << "[sistrip::DigiToRaw::createFedBuffers]"
                     << " strip >= strips_per_fedCh";
                  edm::LogWarning("DigiToRaw") << ss.str();
                  }
                  continue;
                }
                
                // check if value already exists
                if ( edm::isDebugEnabled() ) {
                  const uint16_t value = 0;//chanData[strip];
                  if ( value && value != (*idigi).adc() ) {
                  std::stringstream ss; 
                  ss << "[sistrip::DigiToRaw::createFedBuffers]" 
                     << " Incompatible ADC values in buffer!"
                     << "  FedId/FedCh: " << *ifed << "/" << iconn->fedCh()
                     << "  DetStrip: " << STRIP(idigi, digis_begin)
                     << "  FedChStrip: " << strip
                     << "  AdcValue: " << (*idigi).adc()
                     << "  RawData[" << strip << "]: " << value;
                  edm::LogWarning("DigiToRaw") << ss.str();
                  }
                }
                
                // Add digi to buffer
                chanData[strip] = (*idigi).adc();
              }
            }
            // if ((*idigi).strip() >= (ipair+1)*256) break;
            
            if ( edm::isDebugEnabled() ) {
              edm::LogWarning("DigiToRaw") 
                << "DigiToRaw::createFedBuffers] " 
                << "Almost at the end...";
            }
            //create the buffer
            FEDRawData& fedrawdata = buffers->FEDData( *ifed );
            bufferGenerator_.generateBuffer(&fedrawdata, fedData, *ifed, packetCode_);

            if ( edm::isDebugEnabled() ) {
              std::ostringstream debugStream;
              bufferGenerator_.feHeader().print(debugStream);
              edm::LogWarning("DigiToRaw")
                << "[sistrip::DigiToRaw::createFedBuffers_]"
                << " length of final feHeader: " << bufferGenerator_.feHeader().lengthInBytes() << "\n"
                << debugStream.str();
            }
          }
        }//loop on fedids
        if ( edm::isDebugEnabled() ) {
          edm::LogWarning("DigiToRaw")
            << "[sistrip::DigiToRaw::createFedBuffers_]"
            << "end of first loop on feds";
        }

      }//end of workflow for copying header, below is workflow without copying header
      else{     
        if ( edm::isDebugEnabled() ) {
          edm::LogWarning("DigiToRaw")
            << "[sistrip::DigiToRaw::createFedBuffers_]"
            << "Now getting the digis..."
            << " Number of feds: " << fed_ids.size() << " between "
            << *(fed_ids.begin()) << " and " << *(fed_ids.end());
        }

        for ( auto ifed = fed_ids.begin(); ifed != fed_ids.end(); ++ifed ) {
          
          auto conns = cabling->fedConnections(*ifed);
          
          FEDStripData fedData(dataIsAlready8BitTruncated);
          
          
          for (auto iconn = conns.begin() ; iconn != conns.end(); iconn++ ) {
            
            // Determine FED key from cabling
            uint32_t fed_key = ( ( iconn->fedId() & sistrip::invalid_ ) << 16 ) | ( iconn->fedCh() & sistrip::invalid_ );
            
            // Determine whether DetId or FED key should be used to index digi containers
            uint32_t key = ( useFedKey_ || mode_ == READOUT_MODE_SCOPE ) ? fed_key : iconn->detId();
            
            // Check key is non-zero and valid
            if ( !key || ( key == sistrip::invalid32_ ) ) { continue; }
            
            // Determine APV pair number (needed only when using DetId)
            uint16_t ipair = ( useFedKey_ || mode_ == READOUT_MODE_SCOPE ) ? 0 : iconn->apvPairNumber();
            
            FEDStripData::ChannelData& chanData = fedData[iconn->fedCh()];

            // Find digis for DetID in collection
            if (!collection.isValid()){
              if ( edm::isDebugEnabled() ) {
                edm::LogWarning("DigiToRaw") 
                  << "[DigiToRaw::createFedBuffers] "
                  << "digis collection is not valid...";
              }
              break;
            }
            typename std::vector< edm::DetSet<Digi_t> >::const_iterator digis = collection->find( key );
            if (digis == collection->end()) { continue; } 

            typename edm::DetSet<Digi_t>::const_iterator idigi, digis_begin(digis->data.begin());
            for ( idigi = digis_begin; idigi != digis->data.end(); idigi++ ) {
              
              if ( STRIP(idigi, digis_begin) < ipair*256 ||
                   STRIP(idigi, digis_begin) > ipair*256+255 ) { continue; }
              const unsigned short strip = STRIP(idigi, digis_begin) % 256;
              
              if ( strip >= STRIPS_PER_FEDCH ) {
                if ( edm::isDebugEnabled() ) {
                  std::stringstream ss;
                  ss << "[sistrip::DigiToRaw::createFedBuffers]"
                     << " strip >= strips_per_fedCh";
                  edm::LogWarning("DigiToRaw") << ss.str();
                }
                continue;
              }
              
              // check if value already exists
              if ( edm::isDebugEnabled() ) {
                const uint16_t value = 0;//chanData[strip];
                if ( value && value != (*idigi).adc() ) {
                  std::stringstream ss;
                  ss << "[sistrip::DigiToRaw::createFedBuffers]"
                     << " Incompatible ADC values in buffer!"
                     << "  FedId/FedCh: " << *ifed << "/" << iconn->fedCh()
                     << "  DetStrip: " << STRIP(idigi, digis_begin)
                     << "  FedChStrip: " << strip
                     << "  AdcValue: " << (*idigi).adc()
                     << "  RawData[" << strip << "]: " << value;
                  edm::LogWarning("DigiToRaw") << ss.str();
                }
              }
              
              // Add digi to buffer
              chanData[strip] = (*idigi).adc();
            }
          }
          // if ((*idigi).strip() >= (ipair+1)*256) break;
          
          if ( edm::isDebugEnabled() ) {
            edm::LogWarning("DigiToRaw")
              << "DigiToRaw::createFedBuffers] "
              << "Almost at the end...";
          }
          //create the buffer
          FEDRawData& fedrawdata = buffers->FEDData( *ifed );
          bufferGenerator_.generateBuffer(&fedrawdata, fedData, *ifed, packetCode_);

          if ( edm::isDebugEnabled() ) {
            std::ostringstream debugStream;
            bufferGenerator_.feHeader().print(debugStream);
            edm::LogWarning("DigiToRaw")
              << "[sistrip::DigiToRaw::createFedBuffers_]"
              << " length of final feHeader: " << bufferGenerator_.feHeader().lengthInBytes() << "\n"
              << debugStream.str();
          }
        }//loop on feds
      }//end if-else for copying header
    }//try
    catch (const std::exception& e) {
      if ( edm::isDebugEnabled() ) {
        edm::LogWarning("DigiToRaw")
          << "DigiToRaw::createFedBuffers] "
          << "Exception caught : " << e.what();
      }
    }
  
  }

  inline uint16_t DigiToRaw::STRIP(const edm::DetSet<SiStripDigi>::const_iterator& it, const edm::DetSet<SiStripDigi>::const_iterator& begin) const {return it->strip();}
  inline uint16_t DigiToRaw::STRIP(const edm::DetSet<SiStripRawDigi>::const_iterator& it, const edm::DetSet<SiStripRawDigi>::const_iterator& begin) const {return it-begin;}

}


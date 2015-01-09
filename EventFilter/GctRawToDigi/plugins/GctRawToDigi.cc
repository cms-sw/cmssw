#include "EventFilter/GctRawToDigi/plugins/GctRawToDigi.h"

// System headers
#include <vector>
#include <sstream>
#include <iostream>

// Framework headers
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"

// Raw data collection headers
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

// GCT Format Translators
#include "EventFilter/GctRawToDigi/src/GctFormatTranslateMCLegacy.h"
#include "EventFilter/GctRawToDigi/src/GctFormatTranslateV35.h"
#include "EventFilter/GctRawToDigi/src/GctFormatTranslateV38.h"

// Unpack collections class
#include "EventFilter/GctRawToDigi/src/GctUnpackCollections.h"


// Namespace resolution
using std::cout;
using std::endl;
using std::vector;
using std::string;
using std::dec;
using std::hex;


GctRawToDigi::GctRawToDigi(const edm::ParameterSet& iConfig) :
  inputLabel_(iConfig.getParameter<edm::InputTag>("inputLabel")),
  fedId_(iConfig.getUntrackedParameter<int>("gctFedId", FEDNumbering::MINTriggerGCTFEDID)),
  hltMode_(iConfig.getParameter<bool>("hltMode")),
  numberOfGctSamplesToUnpack_(iConfig.getParameter<unsigned>("numberOfGctSamplesToUnpack")),
  numberOfRctSamplesToUnpack_(iConfig.getParameter<unsigned>("numberOfRctSamplesToUnpack")),
  unpackSharedRegions_(iConfig.getParameter<bool>("unpackSharedRegions")),
  formatVersion_(iConfig.getParameter<unsigned>("unpackerVersion")),
  checkHeaders_(iConfig.getUntrackedParameter<bool>("checkHeaders",false)),
  verbose_(iConfig.getUntrackedParameter<bool>("verbose",false)),
  formatTranslator_(0),
  errors_(0),
  errorCounters_(MAX_ERR_CODE+1),  // initialise with the maximum error codes!
  unpackFailures_(0)
{
  LogDebug("GCT") << "GctRawToDigi will unpack FED Id " << fedId_;

  // If the GctFormatTranslate version has been forced from config file, instantiate the relevant one.
  /***  THIS OBVIOUSLY STINKS - NEED TO REPLACE WITH SOMETHING BETTER THAN MASSIVE IF-ELSE SOON ***/
  /***  WHEN THIS MESS IS REMOVED REMEMBER THAT THE V38 FORMAT TRANSLATE HAS A DIFERENT CTOR TO THE OTHERS ***/
  if(formatVersion_ == 0) { edm::LogInfo("GCT") << "The required GCT Format Translator will be automatically determined from the first S-Link packet header."; }
  else if(formatVersion_ == 1)
  {
    edm::LogInfo("GCT") << "You have selected to use GctFormatTranslateMCLegacy";
    formatTranslator_ = new GctFormatTranslateMCLegacy(hltMode_, unpackSharedRegions_);
  }
  else if(formatVersion_ == 2)
  {
    edm::LogInfo("GCT") << "You have selected to use GctFormatTranslateV35";
    formatTranslator_ = new GctFormatTranslateV35(hltMode_, unpackSharedRegions_);
  }
  else if(formatVersion_ == 3)
  {
    edm::LogInfo("GCT") << "You have selected to use GctFormatTranslateV38";
    formatTranslator_ = new GctFormatTranslateV38(hltMode_, unpackSharedRegions_, numberOfGctSamplesToUnpack_, numberOfRctSamplesToUnpack_);    
  }
  else
  { 
    edm::LogWarning("GCT") << "You have requested a version of GctFormatTranslate that does not exist! Will attempt to auto-detect "
                              "the required GCT Format Translator from the first S-Link packet header instead.";
  }

  if(hltMode_) { edm::LogInfo("GCT") << "HLT unpack mode selected: HLT unpack optimisations will be used."; }
  if(unpackSharedRegions_) { edm::LogInfo("GCT") << "You have selected to unpack shared RCT calo regions - be warned: "
                                                    "this is for commissioning purposes only!"; }

  /** Register Products **/
  // GCT input collections
  produces<L1GctFibreCollection>();
  produces<L1CaloEmCollection>();
  produces<L1CaloRegionCollection>();

  // GCT internal collections
  produces<L1GctInternEmCandCollection>();
  produces<L1GctInternJetDataCollection>();
  produces<L1GctInternEtSumCollection>();
  produces<L1GctInternHFDataCollection>();
  produces<L1GctInternHtMissCollection>();

  // GCT output collections
  produces<L1GctEmCandCollection>("isoEm");
  produces<L1GctEmCandCollection>("nonIsoEm");
  produces<L1GctJetCandCollection>("cenJets");
  produces<L1GctJetCandCollection>("forJets");
  produces<L1GctJetCandCollection>("tauJets");
  produces<L1GctHFBitCountsCollection>();
  produces<L1GctHFRingEtSumsCollection>();
  produces<L1GctEtTotalCollection>();
  produces<L1GctEtHadCollection>();
  produces<L1GctEtMissCollection>();
  produces<L1GctHtMissCollection>();
  produces<L1GctJetCountsCollection>();  // Deprecated (empty collection still needed by GT)
  
  // Error collection
  produces<L1TriggerErrorCollection>();
  consumes<FEDRawDataCollection>(inputLabel_);
}


GctRawToDigi::~GctRawToDigi()
{
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
  delete formatTranslator_;
}

void GctRawToDigi::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("unpackSharedRegions",false);
  desc.add<unsigned int>("numberOfGctSamplesToUnpack",1);
  desc.add<unsigned int>("numberOfRctSamplesToUnpack",1);
  desc.add<bool>("hltMode",false);
  desc.add<edm::InputTag>("inputLabel",edm::InputTag("rawDataCollector"));
  static const char* const kComment=
    " \n"
    "   value   |                        Unpacker/RAW Format Version \n"
    "-----------|---------------------------------------------------------------------------- \n"
    "     0     |   Auto-detects RAW Format in use - the recommended option \n"
    "     1     |   Force usage of the Monte-Carlo Legacy unpacker (unpacks DigiToRaw events) \n"
    "     2     |   Force usage of the RAW Format V35 unpacker \n"
    "     3     |   Force usage of the RAW Format V38 unpacker \n";
  desc.add<unsigned int>("unpackerVersion",0)->setComment(kComment);
  desc.addUntracked<int>("gctFedId",745);
  desc.addUntracked<bool>("checkHeaders",false),
  desc.addUntracked<bool>("verbose",false);
  descriptions.add("gctRawToDigi",desc);
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void GctRawToDigi::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

  // Instantiate all the collections the unpacker needs; puts them in event when this object goes out of scope.
  std::auto_ptr<GctUnpackCollections> colls(new GctUnpackCollections(iEvent));  
  errors_ = colls->errors();
  
  // get raw data collection
  edm::Handle<FEDRawDataCollection> feds;
  iEvent.getByLabel(inputLabel_, feds);

  // if raw data collection is present, do the unpacking
  if (feds.isValid()) {
  
    const FEDRawData& gctRcd = feds->FEDData(fedId_);
 
    LogDebug("GCT") << "Upacking FEDRawData of size " << std::dec << gctRcd.size();

    // check for empty events
    if(gctRcd.size() < 16) {
      LogDebug("GCT") << "Cannot unpack: empty/invalid GCT raw data (size = "
                      << gctRcd.size() << "). Returning empty collections!";
      addError(1);
      return;
    }

    // If no format translator yet set, need to auto-detect from header.
    // If auto format detection fails, we have no concrete format
    // translator instantiated... set error and bail
    if(!formatTranslator_) {
      if(!autoDetectRequiredFormatTranslator(gctRcd.data())) return;
    }
    
    // reset collection of block headers
    blockHeaders_.clear();

    // do the unpacking
    unpack(gctRcd, iEvent, colls.get()); 

    // check headers, if enabled
    if (checkHeaders_) checkHeaders();
    
    // dump summary in verbose mode
    if(verbose_) { doVerboseOutput(blockHeaders_, colls.get()); }
    
  }

}


void GctRawToDigi::unpack(const FEDRawData& d, edm::Event& e, GctUnpackCollections * const colls)
{

  // We should now have a valid formatTranslator pointer
  formatTranslator_->setUnpackCollections(colls);

  const unsigned char * data = d.data();  // The 8-bit wide raw-data array.  

  // Data offset - starts at 16 as there is a 64-bit S-Link header followed
  // by a 64-bit software-controlled header (for pipeline format version
  // info that is not yet used).
  unsigned dPtr = 16;

  const unsigned dEnd = d.size() - 8; // End of payload is at (packet size - final slink header)

  // read blocks
  for (unsigned nb=0; dPtr<dEnd; ++nb)
  {
    if(nb >= MAX_BLOCKS) {
      LogDebug("GCT") << "Reached block limit - bailing out from this event!";
      addError(6);
      break; 
    }
  
    // read block header
    GctBlockHeader blockHeader = formatTranslator_->generateBlockHeader(&data[dPtr]);
  
    // unpack the block; dPtr+4 is to get to the block data.
    if(!formatTranslator_->convertBlock(&data[dPtr+4], blockHeader)) // Record if we had an unpack problem then skip rest of event.
    {
      LogDebug("GCT") << "Encountered block unpack error - bailing out from this event!";
      addError(4);
      break;
    } 

    // advance pointer
    dPtr += 4*(blockHeader.blockLength()*blockHeader.nSamples()+1); // *4 because blockLen is in 32-bit words, +1 for header

    // if verbose or checking block headers, store the header
    if (verbose_ || checkHeaders_) blockHeaders_.push_back(blockHeader);

  }

}


// detect raw data format version from known raw data address 
bool GctRawToDigi::autoDetectRequiredFormatTranslator(const unsigned char * d)
{
  LogDebug("GCT") << "About to auto-detect the required format translator from the firmware version header.";
    
  const uint32_t * p32 = reinterpret_cast<const uint32_t *>(d);
  unsigned firmwareHeader = p32[2];

  /***  THIS OBVIOUSLY STINKS - NEED TO REPLACE WITH SOMETHING BETTER THAN MASSIVE IF-ELSE SOON ***/
  /***  WHEN THIS MESS IS REMOVED REMEMBER THAT THE V38 FORMAT TRANSLATE HAS A DIFERENT CTOR TO THE OTHERS ***/

  if( firmwareHeader >= 25 && firmwareHeader <= 35 )
  {
    edm::LogInfo("GCT") << "Firmware Version V" << firmwareHeader << " detected: GctFormatTranslateV" << firmwareHeader << " will be used to unpack.";
    formatTranslator_ = new GctFormatTranslateV35(hltMode_, unpackSharedRegions_);
    return true;
  }
  else if( firmwareHeader == 38 )
  {
    edm::LogInfo("GCT") << "Firmware Version V" << firmwareHeader << " detected: GctFormatTranslateV" << firmwareHeader << " will be used to unpack.";
    formatTranslator_ = new GctFormatTranslateV38(hltMode_, unpackSharedRegions_, numberOfGctSamplesToUnpack_, numberOfRctSamplesToUnpack_);
    return true;
  }
  else if( firmwareHeader == 0x00000000 )
  {
    edm::LogInfo("GCT") << "Legacy Monte-Carlo data detected: GctFormatTranslateMCLegacy will be used to unpack.";
    formatTranslator_ = new GctFormatTranslateMCLegacy(hltMode_, unpackSharedRegions_);
    return true;
  }
  // these lines comments otherwise error is not reported!!!
  //  else if(firmwareHeader == 0xdeadffff) { /* Driver detected unknown firmware version. L1TriggerError code? */ }
  //  else if( firmwareHeader == 0xaaaaaaaa) { /* Before driver firmware version checks implemented. L1TriggerError code?  */ }
  else { /* Totally unknown firmware header */
  
    LogDebug("GCT") << "Failed to determine unpacker to use from the firmware version header! "
      "(firmware header = 0x" << hex << firmwareHeader << dec << ")";
    addError(2);
    return false;
  }

}


void GctRawToDigi::checkHeaders() {

  // TODO : loop over block headers found this event and check for consistency

}


void GctRawToDigi::doVerboseOutput(const GctBlockHeaderCollection& bHdrs, const GctUnpackCollections * const colls) const
{
  std::ostringstream os;
  os << "Found " << bHdrs.size() << " GCT block headers" << endl;
  for (unsigned i=0, size = bHdrs.size(); i < size; ++i)
  {
    os << "GCT Raw Data Block : " << formatTranslator_->getBlockDescription(bHdrs[i]) << " : " << bHdrs[i] << endl;
  }
  os << *colls << endl;
  edm::LogVerbatim("GCT") << os.str();
}



void GctRawToDigi::addError(const unsigned code) {

  // check this isn't going to break error handling
  if (code > MAX_ERR_CODE) {
    LogDebug("GCT") << "Unknown error code : " << code;
    return;
  }

  // print message on first instance of this error and if verbose flag set to true
  if (errorCounters_.at(code) == 0 && verbose_) {
    std::ostringstream os;
    switch(code) {
      case 0: os << "Reserved error code - not in use"; break;
      case 1: os << "FED record empty or too short"; break;
      case 2: os << "Unknown raw data version"; break;
      case 3: os << "Detected unknown firmware version"; break;
      case 4: os << "Detected unknown data block"; break;
      case 5: os << "Block headers out of sync"; break;
      case 6: os << "Too many blocks"; break;
      default: os << "Unknown error code";
    }
    edm::LogError("GCT") << "Unpacking error " << code << " : " << os.str();
  }

  // increment error counter
  ++(errorCounters_.at(code));
  
  // store error in event if possible
  if (errors_ != 0) {
    errors_->push_back(L1TriggerError(fedId_, code));
  }
  else LogDebug("GCT") << "Detected error (code=" << code << ") but no error collection available!";

}

// ------------ method called once each job just after ending the event loop  ------------
void GctRawToDigi::endJob()
{
  unsigned total=0;
  std::ostringstream os;

  for (unsigned i=0 ; i <= MAX_ERR_CODE ; ++i) {
    total+=errorCounters_.at(i);
    os << "Error " << i << " (" << errorCounters_.at(i) << ")";
    if(i < MAX_ERR_CODE) { os << ", "; }
  }

  if (total>0 && verbose_) {
    edm::LogError("GCT") << "Encountered " << total << " unpacking errors: " << os.str();
  }  
}

/// make this a plugin
DEFINE_FWK_MODULE(GctRawToDigi);

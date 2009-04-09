#include "EventFilter/GctRawToDigi/plugins/GctRawToDigi.h"

// System headers
#include <vector>
#include <sstream>
#include <iostream>

// Framework headers
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"

// Raw data collection headers
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
  fedId_(iConfig.getParameter<int>("gctFedId")),
  hltMode_(iConfig.getParameter<bool>("hltMode")),
  formatVersion_(iConfig.getParameter<unsigned>("unpackerVersion")),
  verbose_(iConfig.getUntrackedParameter<bool>("verbose",false)),
  formatTranslator_(0),
  unpackFailures_(0)
{
  LogDebug("GCT") << "GctRawToDigi will unpack FED Id " << fedId_;

  // If the GctFormatTranslate version has been forced from config file, instantiate the relevant one.
  /***  THIS OBVIOUSLY STINKS - NEED TO REPLACE WITH SOMETHING BETTER THAN MASSIVE IF-ELSE SOON ***/
  if(formatVersion_ == 0) { edm::LogInfo("GCT") << "The required GCT Format Translator will be automatically determined from the first S-Link packet header."; }
  else if(formatVersion_ == 1)
  {
    edm::LogInfo("GCT") << "You have selected to use GctFormatTranslateMCLegacy";
    formatTranslator_ = new GctFormatTranslateMCLegacy(hltMode_);
  }
  else if(formatVersion_ == 2)
  {
    edm::LogInfo("GCT") << "You have selected to use GctFormatTranslateV35";
    formatTranslator_ = new GctFormatTranslateV35(hltMode_);
  }
  else if(formatVersion_ == 3)
  {
    edm::LogInfo("GCT") << "You have selected to use GctFormatTranslateV38";
    formatTranslator_ = new GctFormatTranslateV38(hltMode_);    
  }
  else
  { 
    edm::LogWarning("GCT") << "You have requested a version of GctFormatTranslate that does not exist! Will attempt to auto-detect "
                              "the required GCT Format Translator from the first S-Link packet header instead.";
  }

  if(hltMode_) { edm::LogInfo("GCT") << "HLT unpack mode selected: HLT unpack optimisations will be used."; }


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
}


GctRawToDigi::~GctRawToDigi()
{
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
  delete formatTranslator_;
}


//
// member functions
//

// ------------ method called once each job just before starting event loop  ------------
void GctRawToDigi::beginJob(const edm::EventSetup&)
{
}


// ------------ method called to produce the data  ------------
void GctRawToDigi::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

  // get raw data collection
  edm::Handle<FEDRawDataCollection> feds;
  iEvent.getByLabel(inputLabel_, feds);
  const FEDRawData& gctRcd = feds->FEDData(fedId_);
 
  LogDebug("GCT") << "Upacking FEDRawData of size " << std::dec << gctRcd.size();

  // Instantiate all the collections the unpacker needs; puts them in event when this object goes out of scope.
  std::auto_ptr<GctUnpackCollections> colls(new GctUnpackCollections(iEvent));
  
  // do a simple check of the raw data - this will detect empty events
  if(gctRcd.size() < 16)
  {
    LogDebug("GCT") << "Cannot unpack: empty/invalid GCT raw data (size = "
                    << gctRcd.size() << "). Returning empty collections!";
    ++unpackFailures_;
  }
  else{ unpack(gctRcd, iEvent, colls.get()); }
}


void GctRawToDigi::unpack(const FEDRawData& d, edm::Event& e, GctUnpackCollections * const colls)
{
  const unsigned char * data = d.data();  // The 8-bit wide raw-data array.  

  // If no format translator yet set, need to auto-detect from header.
  if(!formatTranslator_)
  {
    // If auto format detection fails, we have no concrete format
    // translator instantiated... so bail from event.
    if(!autoDetectRequiredFormatTranslator(data)) { return; }
  }

  // We should now have a valid formatTranslator pointer  
  formatTranslator_->setUnpackCollections(colls);

  // Data offset - starts at 16 as there is a 64-bit S-Link header followed
  // by a 64-bit software-controlled header (for pipeline format version
  // info that is not yet used).
  unsigned dPtr = 16;

  const unsigned dEnd = d.size() - 8; // End of payload is at (packet size - final slink header)

  GctBlockHeaderCollection bHdrs; // Vector for storing block headers for verbosity print-out.

  // read blocks
  for (unsigned nb=0; dPtr<dEnd; ++nb)
  {
    if(nb >= MAX_BLOCKS) { LogDebug("GCT") << "Reached block limit - bailing out from this event!"; ++unpackFailures_; break; }
  
    // read block header
    GctBlockHeader blockHeader = formatTranslator_->generateBlockHeader(&data[dPtr]);
  
    // unpack the block; dPtr+4 is to get to the block data.
    if(!formatTranslator_->convertBlock(&data[dPtr+4], blockHeader)) // Record if we had an unpack problem then skip rest of event.
    {
      LogDebug("GCT") << "Encountered block unpack error - bailing out from this event!";
      ++unpackFailures_; break;
    } 

    // advance pointer
    dPtr += 4*(blockHeader.blockLength()*blockHeader.nSamples()+1); // *4 because blockLen is in 32-bit words, +1 for header

    // If verbose, store the header in vector.
    if(verbose_) { bHdrs.push_back(blockHeader); }
  }

  // dump summary in verbose mode
  if(verbose_) { doVerboseOutput(bHdrs, colls); }
}


bool GctRawToDigi::autoDetectRequiredFormatTranslator(const unsigned char * d)
{
  LogDebug("GCT") << "About to auto-detect the required format translator from the firmware version header.";
    
  const uint32_t * p32 = reinterpret_cast<const uint32_t *>(d);
  unsigned firmwareHeader = p32[2];

  /***  THIS OBVIOUSLY STINKS - NEED TO REPLACE WITH SOMETHING BETTER THAN MASSIVE IF-ELSE SOON ***/
  if( firmwareHeader >= 25 && firmwareHeader <= 35 )
  {
    edm::LogInfo("GCT") << "Firmware Version V" << firmwareHeader << " detected: GctFormatTranslateV" << firmwareHeader << " will be used to unpack.";
    formatTranslator_ = new GctFormatTranslateV35(hltMode_);
    return true;
  }
  else if( firmwareHeader == 38 )
  {
    edm::LogInfo("GCT") << "Firmware Version V" << firmwareHeader << " detected: GctFormatTranslateV" << firmwareHeader << " will be used to unpack.";
    formatTranslator_ = new GctFormatTranslateV38(hltMode_);
    return true;
  }
  else if( firmwareHeader == 0x00000000 )
  {
    edm::LogInfo("GCT") << "Legacy Monte-Carlo data detected: GctFormatTranslateMCLegacy will be used to unpack.";
    formatTranslator_ = new GctFormatTranslateMCLegacy(hltMode_);
    return true;
  }
  else if(firmwareHeader == 0xdeadffff) { /* Driver detected unknown firmware version. L1TriggerError code? */ }
  else if( firmwareHeader == 0xaaaaaaaa) { /* Before driver firmware version checks implemented. L1TriggerError code?  */ }
  else { /* Totally unknown firmware header. L1TriggerError code?  */ }
  
  LogDebug("GCT") << "Failed to determine unpacker to use from the firmware version header! "
                     "(firmware header = 0x" << hex << firmwareHeader << dec << ")";

  ++unpackFailures_;
  return false;
}

void GctRawToDigi::doVerboseOutput(const GctBlockHeaderCollection& bHdrs, GctUnpackCollections * const colls)
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

// ------------ method called once each job just after ending the event loop  ------------
void GctRawToDigi::endJob()
{
  if(unpackFailures_ > 0)
  {
    edm::LogError("GCT") << "GCT unpacker encountered " << unpackFailures_
                         << " unpack errors in total during this job!";
  }  
}

/// make this a plugin
DEFINE_FWK_MODULE(GctRawToDigi);

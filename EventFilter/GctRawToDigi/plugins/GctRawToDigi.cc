#include "EventFilter/GctRawToDigi/plugins/GctRawToDigi.h"

// System headers
#include <vector>
#include <sstream>
#include <iostream>

// Framework headers
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"

// Self-tidying vector like boost::ptr_vector.
#include "DataFormats/Common/interface/OwnVector.h"

// Raw data collection headers
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

// GCT raw data format headers
#include "EventFilter/GctRawToDigi/src/GctBlockHeader.h"
#include "EventFilter/GctRawToDigi/src/GctBlockHeaderV2.h"

// GCT input data format headers
#include "DataFormats/L1CaloTrigger/interface/L1CaloEmCand.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"

// GCT output data format headers
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEmCand.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCand.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtSums.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"

// GCT block unpackers
#include "EventFilter/GctRawToDigi/src/GctBlockUnpackerV1.h"
#include "EventFilter/GctRawToDigi/src/GctBlockUnpackerV2.h"

// Namespace resolution
using std::cout;
using std::endl;
using std::vector;


GctRawToDigi::GctRawToDigi(const edm::ParameterSet& iConfig) :
  inputLabel_(iConfig.getParameter<edm::InputTag>("inputLabel")),
  fedId_(iConfig.getParameter<int>("gctFedId")),
  verbose_(iConfig.getUntrackedParameter<bool>("verbose",false)),
  hltMode_(iConfig.getParameter<bool>("hltMode")),
  grenCompatibilityMode_(iConfig.getParameter<bool>("grenCompatibilityMode")),
  blockUnpacker_(0),
  unpackFailures_(0)
{
  LogDebug("GCT") << "GctRawToDigi will unpack FED Id " << fedId_;

  if(grenCompatibilityMode_)
  { 
    edm::LogInfo("GCT") << "GREN 2007 compatibility mode has been selected.";
    blockUnpacker_ = new GctBlockUnpackerV1(hltMode_);
  }
  else { blockUnpacker_ = new GctBlockUnpackerV2(hltMode_); }

  if(hltMode_) { edm::LogInfo("GCT") << "HLT unpack mode selected: HLT unpack optimisations will be used."; }

  //register the products
  produces<L1CaloEmCollection>();
  produces<L1CaloRegionCollection>();
  produces<L1GctEmCandCollection>("isoEm");
  produces<L1GctEmCandCollection>("nonIsoEm");
  produces<L1GctInternEmCandCollection>();
  produces<L1GctJetCandCollection>("cenJets");
  produces<L1GctJetCandCollection>("forJets");
  produces<L1GctJetCandCollection>("tauJets");
  produces<L1GctEtTotalCollection>();
  produces<L1GctEtHadCollection>();
  produces<L1GctEtMissCollection>();
  produces<L1GctHFBitCountsCollection>();
  produces<L1GctHFRingEtSumsCollection>();
  produces<L1GctFibreCollection>();
  produces<L1GctInternJetDataCollection>();
  produces<L1GctInternEtSumCollection>();
  produces<L1GctInternHFDataCollection>();
}


GctRawToDigi::~GctRawToDigi()
{
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
  delete blockUnpacker_;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void GctRawToDigi::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

  // get raw data collection
  edm::Handle<FEDRawDataCollection> feds;
  iEvent.getByLabel(inputLabel_, feds);
  const FEDRawData& gctRcd = feds->FEDData(fedId_);
 
  LogDebug("GCT") << "Upacking FEDRawData of size " << std::dec << gctRcd.size();

  bool invalidDataFlag = false;
  
  // do a simple check of the raw data - this will detect empty events
  if(gctRcd.size() < 16)
  {
      LogDebug("GCT") << "Cannot unpack: empty/invalid GCT raw data (size = "
                      << gctRcd.size() << "). Returning empty collections!";
      invalidDataFlag = true;
  }

  unpack(gctRcd, iEvent, invalidDataFlag);
}


void GctRawToDigi::unpack(const FEDRawData& d, edm::Event& e, const bool invalidDataFlag)
{
  // ** DON'T RESERVE SPACE IN VECTORS FOR DEBUG UNPACK ITEMS! **
  
  // Collections for storing GCT input data.  
  std::auto_ptr<L1CaloEmCollection> rctEm( new L1CaloEmCollection() ); // Input electrons.
  std::auto_ptr<L1CaloRegionCollection> rctCalo( new L1CaloRegionCollection() ); // Input regions.
  
  // GCT output data
  std::auto_ptr<L1GctEmCandCollection>  gctIsoEm   ( new L1GctEmCandCollection() );  gctIsoEm->reserve(4);
  std::auto_ptr<L1GctEmCandCollection>  gctNonIsoEm( new L1GctEmCandCollection() );  gctNonIsoEm->reserve(4);
  std::auto_ptr<L1GctJetCandCollection> gctCenJets ( new L1GctJetCandCollection() ); gctCenJets->reserve(4);
  std::auto_ptr<L1GctJetCandCollection> gctForJets ( new L1GctJetCandCollection() ); gctForJets->reserve(4);
  std::auto_ptr<L1GctJetCandCollection> gctTauJets ( new L1GctJetCandCollection() ); gctTauJets->reserve(4);
  std::auto_ptr<L1GctHFBitCountsCollection> hfBitCounts( new L1GctHFBitCountsCollection() );
  std::auto_ptr<L1GctHFRingEtSumsCollection> hfRingEtSums( new L1GctHFRingEtSumsCollection() );
  std::auto_ptr<L1GctEtTotalCollection> etTotResult( new L1GctEtTotalCollection() );
  std::auto_ptr<L1GctEtHadCollection> etHadResult( new L1GctEtHadCollection() );
  std::auto_ptr<L1GctEtMissCollection> etMissResult( new L1GctEtMissCollection() );

  // GCT intermediate data
  std::auto_ptr<L1GctInternEmCandCollection> gctInternEm( new L1GctInternEmCandCollection() ); 
  std::auto_ptr<L1GctInternJetDataCollection> gctInternJets( new L1GctInternJetDataCollection() ); 
  std::auto_ptr<L1GctInternEtSumCollection> gctInternEtSums( new L1GctInternEtSumCollection() ); 
  std::auto_ptr<L1GctInternHFDataCollection> gctInternHFData( new L1GctInternHFDataCollection() ); 

  // Fibres
  std::auto_ptr<L1GctFibreCollection> gctFibres( new L1GctFibreCollection() );

  if(invalidDataFlag == false) // Only attempt unpack with valid data
  {

    blockUnpacker_->setIsoEmCollection( gctIsoEm.get() );
    blockUnpacker_->setNonIsoEmCollection( gctNonIsoEm.get() );
    blockUnpacker_->setCentralJetCollection( gctCenJets.get() );
    blockUnpacker_->setForwardJetCollection( gctForJets.get() );
    blockUnpacker_->setTauJetCollection( gctTauJets.get() );
    blockUnpacker_->setHFBitCountsCollection( hfBitCounts.get() );
    blockUnpacker_->setHFRingEtSumsCollection( hfRingEtSums.get() );
    blockUnpacker_->setEtTotalCollection( etTotResult.get() );
    blockUnpacker_->setEtHadCollection( etHadResult.get() );
    blockUnpacker_->setEtMissCollection( etMissResult.get() );
    blockUnpacker_->setRctEmCollection( rctEm.get() );
    blockUnpacker_->setRctCaloRegionCollection( rctCalo.get() );
    blockUnpacker_->setInternEmCollection( gctInternEm.get() );
    blockUnpacker_->setInternJetDataCollection( gctInternJets.get() );
    blockUnpacker_->setInternEtSumCollection( gctInternEtSums.get() );
    blockUnpacker_->setInternHFDataCollection( gctInternHFData.get() );
    blockUnpacker_->setFibreCollection( gctFibres.get() );
  
    const unsigned char * data = d.data();  // The 8-bit wide raw-data array.  

    // Data offset - starts at 16 as there is a 64-bit S-Link header followed
    // by a 64-bit software-controlled header (for pipeline format version
    // info that is not yet used).
    unsigned dPtr = 16;
    
    if(grenCompatibilityMode_) { dPtr = 8; }  // No software-controlled secondary header in old scheme. 
    
    const unsigned dEnd = d.size() - 8; // End of payload is at (packet size - final slink header)

    edm::OwnVector<GctBlockHeaderBase> bHdrs; // Self-cleaning vector for storing block headers for verbosity print-out.

    // read blocks
    for (unsigned nb=0; dPtr<dEnd; ++nb)
    {
      if(nb >= MAX_BLOCKS) { LogDebug("GCT") << "Reached block limit - bailing out from this event!"; ++unpackFailures_; break; }
      
      // read block header
      std::auto_ptr<GctBlockHeaderBase> blockHeader;
      if(grenCompatibilityMode_) { blockHeader = std::auto_ptr<GctBlockHeaderBase>(new GctBlockHeader(&data[dPtr])); }
      else { blockHeader = std::auto_ptr<GctBlockHeaderBase>(new GctBlockHeaderV2(&data[dPtr])); }
      
      // unpack the block; dPtr+4 is to get to the block data.
      if(!blockUnpacker_->convertBlock(&data[dPtr+4], *blockHeader)) // Record if we had an unpack problem then skip rest of event.
      {
        LogDebug("GCT") << "Encountered block unpack error - bailing out from this event!";
        ++unpackFailures_; break;
      } 
  
      // advance pointer
      dPtr += 4*(blockHeader->length()*blockHeader->nSamples()+1); // *4 because blockLen is in 32-bit words, +1 for header

      // If verbose, store the header in vector.
      if(verbose_) { bHdrs.push_back(blockHeader); }
    }
  
    // dump summary in verbose mode
    if (verbose_)
    {
      std::ostringstream os;
      os << "Found " << bHdrs.size() << " GCT block headers" << endl;
      for (unsigned i=0, size = bHdrs.size(); i<size; ++i) { os << bHdrs[i]<< endl; }
      os << "Read " << rctEm->size() << " RCT EM candidates" << endl;
      os << "Read " << rctCalo->size() << " RCT Calo Regions" << endl;
      os << "Read " << gctIsoEm->size() << " GCT iso EM candidates" << endl;
      os << "Read " << gctNonIsoEm->size() << " GCT non-iso EM candidates" << endl;
      os << "Read " << gctInternEm->size() << " GCT intermediate EM candidates" << endl;
      os << "Read " << gctCenJets->size() << " GCT central jet candidates" << endl;
      os << "Read " << gctForJets->size() << " GCT forward jet candidates" << endl;
      os << "Read " << gctTauJets->size() << " GCT tau jet candidates" << endl;
      os << "Read " << gctInternJets->size() << " GCT intermediate jet candidates" << endl;
      os << "Read " << etTotResult->size() << " GCT total et" << endl;
      os << "Read " << etHadResult->size() << " GCT ht" << endl;
      os << "Read " << etMissResult->size() << " GCT met" << endl;
      os << "Read " << gctInternEtSums->size() << " GCT intermediate et sums" << endl;
      os << "Read " << hfRingEtSums->size() << " GCT HF ring et sums" << endl;
      os << "Read " << hfBitCounts->size() << " GCT HF ring bit counts" << endl;
      os << "Read " << gctInternHFData->size() << " GCT intermediate HF data" << endl;
      os << "Read " << gctFibres->size() << " GCT raw fibre data" << endl;
      edm::LogVerbatim("GCT") << os.str();
    }
  }
  else { ++unpackFailures_; }

  // put data into the event
  e.put(gctIsoEm, "isoEm");
  e.put(gctNonIsoEm, "nonIsoEm");
  e.put(gctCenJets,"cenJets");
  e.put(gctForJets,"forJets");
  e.put(gctTauJets,"tauJets");
  e.put(hfBitCounts);
  e.put(hfRingEtSums);
  e.put(etTotResult);
  e.put(etHadResult);
  e.put(etMissResult);
  e.put(gctInternEm);
  e.put(gctInternJets);
  e.put(gctInternEtSums);
  e.put(gctInternHFData);
  e.put(rctEm);
  e.put(rctCalo);
  e.put(gctFibres);

}


// ------------ method called once each job just before starting event loop  ------------
void 
GctRawToDigi::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
GctRawToDigi::endJob()
{
  if(unpackFailures_ > 0)
  {
    edm::LogError("GCT") << "GCT unpacker encountered " << unpackFailures_
                         << " unpack errors in total during this job!";
  }  
}



/// make this a plugin
DEFINE_FWK_MODULE(GctRawToDigi);


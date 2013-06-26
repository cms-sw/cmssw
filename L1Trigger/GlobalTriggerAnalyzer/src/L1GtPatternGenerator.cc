/**
 * \class L1GtPatternGenerator
 * 
 * 
 * Description: see header file.  
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Thomas Themel - HEPHY Vienna
 * 
 * $Date$
 * $Revision$
 *
 */

// this class header
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtPatternGenerator.h"

// system include files
#include <memory>
#include <iomanip>
#include <fstream>

// user include files
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctHtMiss.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtFdlWord.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtUtils.h"
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtPatternMap.h"
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtPatternWriter.h"

// constructor
L1GtPatternGenerator::L1GtPatternGenerator(const edm::ParameterSet& parSet)
{
    // input tags for trigger records
    m_gctTag = parSet.getParameter<edm::InputTag>("GctInputTag");
    m_gmtTag = parSet.getParameter<edm::InputTag>("GmtInputTag");
    m_gtTag = parSet.getParameter<edm::InputTag>("GtInputTag");
    m_dtTag = parSet.getParameter<edm::InputTag>("DtInputTag");
    m_cscTag = parSet.getParameter<edm::InputTag>("CscInputTag");
    m_rpcbTag = parSet.getParameter<edm::InputTag>("RpcbInputTag");
    m_rpcfTag = parSet.getParameter<edm::InputTag>("RpcfInputTag");

    // output formatting stuff
    m_header = parSet.getParameter<std::string>("PatternFileHeader");
    m_footer = parSet.getParameter<std::string>("PatternFileFooter");
    m_columnNames = parSet.getParameter<std::vector<std::string> >("PatternFileColumns");
    m_columnLengths = parSet.getParameter<std::vector<uint32_t> >("PatternFileLengths");
    m_columnDefaults = parSet.getParameter<std::vector<uint32_t> >("PatternFileDefaultValues");
    m_fileName = parSet.getParameter<std::string>("PatternFileName");
    m_bx = parSet.getParameter<std::vector<int> >("bx");
    m_debug = parSet.getParameter<bool>("DebugOutput");


    if(m_columnLengths.size() != m_columnNames.size()) { 
      edm::LogWarning("L1GtPatternGenerator") 
	<< "Length of PatternFileColumns does not match length of PatternFileLenghts, " <<
	m_columnNames.size() << " vs " << m_columnLengths.size() << std::endl;
    }

    LogDebug("L1GtPatternGenerator")
      << "\nL1 GCT  record:            "
      << m_gctTag 
      << "\nL1 GMT record:             "
      << m_gmtTag
      << "\nL1 GT record:              "
      << m_gtTag << std::endl;
}

// destructor
L1GtPatternGenerator::~L1GtPatternGenerator()
{}

// local helper functions

/** Actual data extraction. The template parameters are neccessary because we have to handle
    multiple unrelated record types and call member functions of varying return type, but note
    that they don't need to be specified in practice because the rawFunctionPtr parameter defines
    them anyway. 

    @param iEvent          The event to get the records from.
    @param allPatterns     Destination object
    @param label           First half of the input tag that identifies our source in the event
    @param instance        Second half of the input tag that identifies our source in the event
    @param rawFunctionPtr  Pointer-to-member-function that specifies a getter function that extracts
                           the information we want from the record object. 
    @param prefix          The column name prefix that defines how the columns in the pattern map will
                           be named (@see L1GtPatternLine::push)
    @param packingFunction an optional function that the raw value is passed through befor it is 
                           added to the pattern line
*/
template <class TRecord, typename TResult> static void extractRecordData(const edm::Event& iEvent,
									 L1GtPatternMap& allPatterns,
									 const std::string& label, 
									 const std::string& instance, 
									 TResult (TRecord::*rawFunctionPtr)() const,
									 const std::string& prefix,
                                                                         uint32_t (*packingFunction)(uint32_t) = NULL)
{
  uint32_t valueCount;

  // Extract record from event.
  edm::Handle<std::vector<TRecord> > handle;
  iEvent.getByLabel(label, instance, handle);

  if(!handle.isValid()) {
    throw cms::Exception(__func__) << "Failed to extract record of type " << typeid(TRecord).name() <<
      " labeled " << label << ", instance " << instance;
  }

  edm::EventNumber_t eventNr = iEvent.id().event();

  // Then loop over collection and add each event to the map.
  for(typename std::vector<TRecord>::const_iterator it = handle->begin(); it != handle->end(); ++it) {
    int bx = it->bx();
    L1GtPatternLine& line = allPatterns.getLine(eventNr, bx);
    uint32_t value = ((*it).*rawFunctionPtr)();
    if(packingFunction != NULL) { 
        value = packingFunction(value);
    } 

    line.push(prefix, value);
    ++valueCount;
  }
}

/*** Convert a vector of bools into a vector of uint32_ts. Probably
     optimizable, but let's just trust that it doesn't matter... */
static std::vector<uint32_t> chopWords(const std::vector<bool>& aWord) {
  std::vector<uint32_t> result;

  result.resize((aWord.size()+31)/32, 0);

  for(unsigned i = 0 ; i < aWord.size(); ++i) {
    result[i/32] |= aWord[i] << (i%32);
  }

  return result;
}

/** Split a vector<bool> of arbitrary size into uint32_t chunks and add them to a
    pattern file line with the given prefix.
*/
static void extractGlobalTriggerWord(const std::vector<bool> input, L1GtPatternLine& line, const std::string& prefix)
{  
  std::vector<uint32_t> resultWords = chopWords(input);

  // add in reverse order, so that higher-order words have lower indices 
  // (like in "natural" number representation) (10 -> digit1=1 digit2=0)
  for(unsigned i = resultWords.size() ; i > 0; --i) {
    line.push(prefix, resultWords[i-1]);
  }
}

/** Bits 8..15 (5 bits Pt, 3 bits quality) need to be inverted on the GMT inputs.
 *  See http://wwwhephy.oeaw.ac.at/p3w/cms/trigger/globalMuonTrigger/notes/in04_022.pdf
 */
uint32_t L1GtPatternGenerator::packRegionalMuons(uint32_t rawData) {
    uint32_t invertMask = 0x0000FF00;
    uint32_t toKeep =   rawData & (~invertMask);
    return toKeep | (~rawData & invertMask);
}


// member functions
void L1GtPatternGenerator::extractGlobalTriggerData(const edm::Event& iEvent, L1GtPatternMap& patterns) {

  // extract global trigger readout record
  edm::Handle<L1GlobalTriggerReadoutRecord> handle;
  iEvent.getByLabel(m_gtTag, handle);

  // continue if it's present
  if(!handle.isValid()) { 
    throw cms::Exception(__func__) << "Failed to extract GT readout record labeled " 
				   << m_gtTag.label() << ", instance " << m_gtTag.instance();
  }

  edm::EventNumber_t eventNr = iEvent.id().event();

  // for each FDL word...
  const std::vector<L1GtFdlWord>& fdlWords = handle->gtFdlVector();
  for(std::vector<L1GtFdlWord>::const_iterator it = fdlWords.begin();
      it != fdlWords.end() ; ++it) { 
    // extract relevant data
    int bx = it->bxInEvent();

    // find matching pattern file line
    L1GtPatternLine& line = patterns.getLine(eventNr, bx);

    extractGlobalTriggerWord(it->gtDecisionWord(), line, "gtDecision");
    extractGlobalTriggerWord(it->gtDecisionWordExtended(), line, "gtDecisionExt");
    extractGlobalTriggerWord(it->gtTechnicalTriggerWord(), line, "gtTechTrigger");

    line.push("gtFinalOr", it->finalOR());
  }
}

/** The mapping from hfBitCounts/hfRingEtSums raw data to the PSBs is non-trivial, see
 *  http://wwwhephy.oeaw.ac.at/p3w/electronic1/GlobalTrigger/doc/InterfaceDesc/update_CMS_NOTE_2002_069.pdf
 */
void L1GtPatternGenerator::packHfRecords(const std::string& resultName, L1GtPatternMap& allPatterns)
{
  // iterate over each pattern line
  for(L1GtPatternMap::LineMap::iterator it = allPatterns.begin(); 
      it != allPatterns.end(); ++it) { 
    // Get the HF bit counts and ring sums 
    uint32_t counts = it->second.get("hfBitCounts1");
    uint32_t sums = it->second.get("hfRingEtSums1");
    
    
    // Bits 0..11 -> 4 bit counts
    uint32_t hfPsbValue = (counts & 0xFFF) |
        // Bit 12..14 ring 1 pos. rap. HF Et sum
        (sums & 0x7) << 12 | 
        // Bits 16.. rest of the ET sums
        (sums >> 3) << 16;
    // TODO: Spec states non-data values for Bits 15, 31, 47 and 63.

    // Export computed value to pattern writer. */
    it->second.push(resultName, hfPsbValue);
  }
}

/** Analyze each event: 
    - Extract the input records that interest us from the event
    - Split them into pattern file lines according to their bx number
    - Format the lines and write them to the file. 
*/
void L1GtPatternGenerator::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup)
{
    // debug information
    const unsigned int runNumber = iEvent.run();
    const unsigned int lsNumber = iEvent.luminosityBlock();
    const unsigned int eventNumber = iEvent.id().event();

    LogTrace("L1GtPatternGenerator") << "\n\nL1GtPatternGenerator::analyze: Run: " << runNumber << " LS: " << lsNumber << " Event: "
            << eventNumber << "\n\n" << std::endl;
    

  L1GtPatternMap allPatterns;

  // GMT muon candidates
  extractRecordData(iEvent, allPatterns, m_gmtTag.label(), m_gmtTag.instance(), &L1MuGMTCand::getDataWord, "gmtMuon");

  // regional muon candidates
  extractRecordData(iEvent, allPatterns, m_cscTag.label(), m_cscTag.instance(), &L1MuRegionalCand::getDataWord, "cscMuon", packRegionalMuons);  
  extractRecordData(iEvent, allPatterns, m_dtTag.label(),  m_dtTag.instance(), &L1MuRegionalCand::getDataWord, "dtMuon", packRegionalMuons);
  extractRecordData(iEvent, allPatterns, m_rpcfTag.label(), m_rpcfTag.instance(), &L1MuRegionalCand::getDataWord, "fwdMuon", packRegionalMuons);
  extractRecordData(iEvent, allPatterns, m_rpcbTag.label(), m_rpcbTag.instance(), &L1MuRegionalCand::getDataWord, "brlMuon", packRegionalMuons);

  // GCT objects
  extractRecordData(iEvent, allPatterns, m_gctTag.label(), "nonIsoEm", &L1GctEmCand::raw, "gctEm");
  extractRecordData(iEvent, allPatterns, m_gctTag.label(), "isoEm", &L1GctEmCand::raw, "gctIsoEm");
  extractRecordData(iEvent, allPatterns, m_gctTag.label(), "", &L1GctEtMiss::et, "etMiss");
  extractRecordData(iEvent, allPatterns, m_gctTag.label(), "", &L1GctEtMiss::phi, "etMissPhi");
  extractRecordData(iEvent, allPatterns, m_gctTag.label(), "", &L1GctHtMiss::raw, "htMiss");
  extractRecordData(iEvent, allPatterns, m_gctTag.label(), "", &L1GctEtHad::raw, "etHad");
  extractRecordData(iEvent, allPatterns, m_gctTag.label(), "", &L1GctEtTotal::raw, "etTotal");
  extractRecordData(iEvent, allPatterns, m_gctTag.label(), "cenJets", &L1GctJetCand::raw, "cenJet");
  extractRecordData(iEvent, allPatterns, m_gctTag.label(), "forJets", &L1GctJetCand::raw, "forJet");
  extractRecordData(iEvent, allPatterns, m_gctTag.label(), "tauJets", &L1GctJetCand::raw, "tauJet");
  extractRecordData(iEvent, allPatterns, m_gctTag.label(), "", &L1GctHFBitCounts::raw, "hfBitCounts");
  extractRecordData(iEvent, allPatterns, m_gctTag.label(), "", &L1GctHFRingEtSums::raw, "hfRingEtSums");

  // Post processing: 
  // HFBitCounts/HFRingEtSums need to be mangled to PSB values
  packHfRecords("hfPsbValue", allPatterns);

  // GT objects
  extractGlobalTriggerData(iEvent, allPatterns);

  // Output
  m_writer->writePatterns(allPatterns);
}

/** Method called once each job just before starting event loop.
    - Initialize the output file and the writer object. 
*/
void L1GtPatternGenerator::beginJob()
{
  m_fileStream.open(m_fileName.c_str());

  if(!m_fileStream) { 
    edm::LogError("L1GtPatternGenerator") <<  "Failed to open output file " << m_fileName;
  }

  m_writer = std::auto_ptr<L1GtPatternWriter>(new L1GtPatternWriter(m_fileStream, m_header, m_footer, m_columnNames, m_columnLengths, m_columnDefaults, m_bx, m_debug));
}

/** Method called once each job just after ending the event loop.
    - Close the output file stream.
 */
void L1GtPatternGenerator::endJob()
{
  m_writer->close();
  m_fileStream.close();
}

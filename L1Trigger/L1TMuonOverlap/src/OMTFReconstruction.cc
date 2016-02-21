#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"

#include "CondFormats/DataRecord/interface/L1TMuonOverlapParamsRcd.h"
#include "CondFormats/L1TObjects/interface/L1TMuonOverlapParams.h"

#include "L1Trigger/L1TMuonOverlap/plugins/L1TMuonOverlapTrackProducer.h"
#include "L1Trigger/L1TMuonOverlap/interface/OMTFProcessor.h"
#include "L1Trigger/L1TMuonOverlap/interface/OMTFinput.h"
#include "L1Trigger/L1TMuonOverlap/interface/OMTFReconstruction.h"
#include "L1Trigger/L1TMuonOverlap/interface/OMTFConfiguration.h"
#include "L1Trigger/L1TMuonOverlap/interface/XMLConfigWriter.h"

#include "L1Trigger/RPCTrigger/interface/RPCConst.h"

OMTFReconstruction::OMTFReconstruction() :
  m_OMTFConfig(0), m_OMTF(0), aTopElement(0), m_OMTFConfigMaker(0), m_Writer(0){}

OMTFReconstruction::OMTFReconstruction(const edm::ParameterSet& theConfig) :
  m_Config(theConfig), m_OMTFConfig(0), m_OMTF(0), aTopElement(0), m_OMTFConfigMaker(0), m_Writer(0) {


  if(!m_Config.exists("omtf")){
    edm::LogError("L1TMuonOverlapTrackProducer")<<"omtf configuration not found in cfg.py";
  }

  dumpResultToXML = m_Config.getParameter<bool>("dumpResultToXML");
  dumpDetailedResultToXML = m_Config.getParameter<bool>("dumpDetailedResultToXML");
  m_Config.getParameter<std::string>("XMLDumpFileName");

  if(dumpResultToXML){
    m_Writer = new XMLConfigWriter();
    std::string fName = "OMTF";
    m_Writer->initialiseXMLDocument(fName);
  }
}

/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
OMTFReconstruction::~OMTFReconstruction(){
  
  delete m_OMTFConfig;
  delete m_OMTF;  

  if (m_Writer) delete m_Writer;
}

/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
void OMTFReconstruction::beginJob() {
  
  if(m_Config.exists("omtf")){
    m_OMTFConfig = new OMTFConfiguration(m_Config.getParameter<edm::ParameterSet>("omtf"));
    m_OMTF = new OMTFProcessor(m_Config.getParameter<edm::ParameterSet>("omtf"));
  }
}

/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
void OMTFReconstruction::endJob(){

  if(dumpResultToXML){
    std::string fName = m_Config.getParameter<std::string>("XMLDumpFileName");
    m_Writer->finaliseXMLDocument(fName);
  } 
}

/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
void OMTFReconstruction::beginRun(edm::Run const& run, edm::EventSetup const& iSetup) {

  ///If configuration is read from XML do not look at the DB.
  if(m_Config.getParameter<edm::ParameterSet>("omtf").getParameter<bool>("configFromXML")) return;  

  const L1TMuonOverlapParamsRcd& omtfParamsRcd = iSetup.get<L1TMuonOverlapParamsRcd>();
  
  edm::ESHandle<L1TMuonOverlapParams> omtfParamsHandle;
  omtfParamsRcd.get(omtfParamsHandle);

  const L1TMuonOverlapParams* omtfParams = omtfParamsHandle.product();
  if (!omtfParams) {
    edm::LogError("L1TMuonOverlapTrackProducer") << "Could not retrieve parameters from Event Setup" << std::endl;
  }

  m_OMTFConfig->configure(omtfParams);
  m_OMTF->configure(omtfParams);
}

void OMTFReconstruction::algoBeginJob() {

  m_OMTF = new OMTFProcessor(m_Config.getParameter<edm::ParameterSet>("omtf"));
}

/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
AlgoMuonResults OMTFReconstruction::algoReconstruct(const edm::Event& iEvent, const edm::EventSetup& evSetup) {

  const L1TMuonOverlapParamsRcd& omtfParamsRcd = evSetup.get<L1TMuonOverlapParamsRcd>();
  
  edm::ESHandle<L1TMuonOverlapParams> omtfParamsHandle;
  omtfParamsRcd.get(omtfParamsHandle);

  const L1TMuonOverlapParams* omtfParams = omtfParamsHandle.product();
  if (!omtfParams) {
    edm::LogError("L1TMuonOverlapTrackProducer") << "Could not retrieve parameters from Event Setup" << std::endl;
  }

  m_OMTF = new OMTFProcessor(m_Config.getParameter<edm::ParameterSet>("omtf"));
  m_OMTF->configure(omtfParams);

  m_InputMaker.initialize(evSetup);

  edm::Handle<L1MuDTChambPhContainer> dtPhDigis;
  edm::Handle<L1MuDTChambThContainer> dtThDigis;
  edm::Handle<CSCCorrelatedLCTDigiCollection> cscDigis;
  edm::Handle<RPCDigiCollection> rpcDigis;

  if(!m_Config.getParameter<bool>("dropDTPrimitives")){
    iEvent.getByLabel(m_Config.getParameter<edm::InputTag>("srcDTPh"),dtPhDigis);
    iEvent.getByLabel(m_Config.getParameter<edm::InputTag>("srcDTTh"),dtThDigis);
  }
  if(!m_Config.getParameter<bool>("dropRPCPrimitives")) iEvent.getByLabel(m_Config.getParameter<edm::InputTag>("srcRPC"),rpcDigis);
  if(!m_Config.getParameter<bool>("dropCSCPrimitives")) iEvent.getByLabel(m_Config.getParameter<edm::InputTag>("srcCSC"),cscDigis);

  AlgoMuonResults algoMuonResults;
  algoMuonResults.clear();

  ///Loop over all processors, each covering 60 deg in phi
  for(unsigned int iProcessor=0;iProcessor<6;++iProcessor){

    OMTFinput inputPos = m_InputMaker.buildInputForProcessor(dtPhDigis.product(),
                         dtThDigis.product(),
                         cscDigis.product(),
                         rpcDigis.product(),
                         iProcessor,
                         l1t::tftype::omtf_pos);
    OMTFinput inputNeg = m_InputMaker.buildInputForProcessor(dtPhDigis.product(),
                         dtThDigis.product(),
                         cscDigis.product(), 
                         rpcDigis.product(),
                         iProcessor, 
                         l1t::tftype::omtf_neg);

    std::vector<AlgoMuon> algoCandidatesPos, algoCandidatesNeg;

    const std::vector<OMTFProcessor::resultsMap> & resultsPos = m_OMTF->processInput(iProcessor, inputPos);
    m_Sorter.sortRefHitResults(resultsPos, algoCandidatesPos);  
    algoMuonResults.push_back(make_pair(iProcessor*2, algoCandidatesPos));

    const std::vector<OMTFProcessor::resultsMap> & resultsNeg = m_OMTF->processInput(iProcessor, inputNeg);
    m_Sorter.sortRefHitResults(resultsNeg, algoCandidatesNeg);        
    algoMuonResults.push_back(make_pair(iProcessor*2 + 1, algoCandidatesNeg));
  }

  delete m_OMTF;
  return algoMuonResults; 
}


/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
std::auto_ptr<l1t::RegionalMuonCandBxCollection > OMTFReconstruction::reconstruct(const edm::Event& iEvent, const edm::EventSetup& evSetup) {

  m_InputMaker.initialize(evSetup); //FIXME shoun't it be in beginRun?

  edm::Handle<L1MuDTChambPhContainer> dtPhDigis;
  edm::Handle<L1MuDTChambThContainer> dtThDigis;
  edm::Handle<CSCCorrelatedLCTDigiCollection> cscDigis;
  edm::Handle<RPCDigiCollection> rpcDigis;

  ///Filter digis by dropping digis from selected (by cfg.py) subsystems
  if(!m_Config.getParameter<bool>("dropDTPrimitives")){
    iEvent.getByLabel(m_Config.getParameter<edm::InputTag>("srcDTPh"),dtPhDigis);
    iEvent.getByLabel(m_Config.getParameter<edm::InputTag>("srcDTTh"),dtThDigis);
  }
  if(!m_Config.getParameter<bool>("dropRPCPrimitives")) iEvent.getByLabel(m_Config.getParameter<edm::InputTag>("srcRPC"),rpcDigis);
  if(!m_Config.getParameter<bool>("dropCSCPrimitives")) iEvent.getByLabel(m_Config.getParameter<edm::InputTag>("srcCSC"),cscDigis);

    std::auto_ptr<l1t::RegionalMuonCandBxCollection > candidates(new l1t::RegionalMuonCandBxCollection);
    if(dumpResultToXML) aTopElement = m_Writer->writeEventHeader(iEvent.id().event());

    // NOTE: assuming all is for bx 0
    int bx = 0;
    m_AlgoMuonResults.clear();

    ///Loop over all processors, each covering 60 deg in phi
    for(unsigned int iProcessor=0;iProcessor<6;++iProcessor){

      ///Input data with phi ranges shifted for each processor, so it fits 11 bits range
      OMTFinput inputPos = m_InputMaker.buildInputForProcessor(dtPhDigis.product(),
                         dtThDigis.product(),
                         cscDigis.product(),
                         rpcDigis.product(),
                         iProcessor,
                         l1t::tftype::omtf_pos);

      OMTFinput inputNeg = m_InputMaker.buildInputForProcessor(dtPhDigis.product(),
                         dtThDigis.product(),
                         cscDigis.product(), 
                         rpcDigis.product(),
                         iProcessor, 
                         l1t::tftype::omtf_neg);
            
      l1t::RegionalMuonCandBxCollection OTFCandidatesPos, OTFCandidatesNeg;
      std::vector<AlgoMuon> algoCandidatesPos, algoCandidatesNeg;
      
      //====================
      // positive
      //====================
      ///Results for each GP in each logic region of given processor
      //Retreive all candidates returned by sorter: upto 3 non empty ones with different phi or charge
      const std::vector<OMTFProcessor::resultsMap> & resultsPos = m_OMTF->processInput(iProcessor, inputPos);
      m_Sorter.sortRefHitResults(resultsPos, algoCandidatesPos);  
      m_AlgoMuonResults.push_back(make_pair(iProcessor*2, algoCandidatesPos));
      // m_GhostBuster.select(algoCandidatesPos); 
      m_Sorter.rewriteToRegionalMuon(algoCandidatesPos, OTFCandidatesPos, bx);  // rewrite AlgoMuon to RegionalMuon
      ///Shift phi scales, and put MicroGMT candidates into candidates collection
      m_Sorter.processCandidates(iProcessor, bx, candidates, OTFCandidatesPos, l1t::tftype::omtf_pos);

      //====================
      // negative
      //====================
      const std::vector<OMTFProcessor::resultsMap> & resultsNeg = m_OMTF->processInput(iProcessor, inputNeg);
      m_Sorter.sortRefHitResults(resultsNeg, algoCandidatesNeg);        
      m_AlgoMuonResults.push_back(make_pair(iProcessor*2 + 1, algoCandidatesNeg));
      // m_GhostBuster.select(algoCandidatesNeg); 
      m_Sorter.rewriteToRegionalMuon(algoCandidatesNeg, OTFCandidatesNeg, bx);  // rewrite AlgoMuon to RegionalMuon
      ///Shift phi scales, and put MicroGMT candidates into m_Cands collection
      m_Sorter.processCandidates(iProcessor, bx, candidates, OTFCandidatesNeg, l1t::tftype::omtf_neg);

      ///Write data to XML file
      // if(dumpResultToXML){
      //   xercesc::DOMElement * aProcElement = m_Writer->writeEventData(aTopElement,iProcessor,inputPos);
      //   for(unsigned int iRefHit=0; iRefHit < OMTFConfiguration::nTestRefHits; ++iRefHit){
      //     ///Dump only regions, where a candidate was found
      //     // AlgoMuon m_Cand = m_Sorter.sortRefHitResults(resultsPos[iRefHit],0);//charge=0 means ignore charge
      //     AlgoMuon algoCand = algoCandidatesPos[iRefHit];          
      //     if(algoCand.getPt()){
      //       m_Writer->writeCandidateData(aProcElement, iRefHit, algoCand);
      //       if(dumpDetailedResultToXML){
      //         for(auto & itKey: resultsNeg[iRefHit]) m_Writer->writeResultsData(aProcElement,
      //                       iRefHit,
      //                       itKey.first,itKey.second);
      //       }
      //     }
      //   }
      // }
      
    }
    return candidates;
  }

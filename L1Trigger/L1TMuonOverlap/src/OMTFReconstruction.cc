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
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
OMTFReconstruction::OMTFReconstruction(const edm::ParameterSet& theConfig) :
  m_Config(theConfig), m_OMTFConfig(0), m_OMTF(0), aTopElement(0), m_OMTFConfigMaker(0), m_Writer(0) {

  dumpResultToXML = m_Config.getParameter<bool>("dumpResultToXML");
  dumpDetailedResultToXML = m_Config.getParameter<bool>("dumpDetailedResultToXML");
  m_Config.getParameter<std::string>("XMLDumpFileName");  
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
  
    m_OMTFConfig = new OMTFConfiguration();
    m_OMTF = new OMTFProcessor();

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

  const L1TMuonOverlapParamsRcd& omtfRcd = iSetup.get<L1TMuonOverlapParamsRcd>();
  
  edm::ESHandle<L1TMuonOverlapParams> omtfParamsHandle;
  omtfRcd.get(omtfParamsHandle);

  const L1TMuonOverlapParams* omtfParams = omtfParamsHandle.product();

  if (!omtfParams) {
    edm::LogError("L1TMuonOverlapTrackProducer") << "Could not retrieve parameters from Event Setup" << std::endl;
  }

  m_OMTFConfig->configure(omtfParams);
  m_OMTF->configure(m_OMTFConfig, omtfParams);
  m_GhostBuster.setNphiBins(m_OMTFConfig->nPhiBins());
  m_Sorter.setNphiBins(m_OMTFConfig->nPhiBins());

  m_InputMaker.initialize(iSetup, m_OMTFConfig);

  if(dumpResultToXML){
    m_Writer = new XMLConfigWriter(m_OMTFConfig);
    std::string fName = "OMTF";
    m_Writer->initialiseXMLDocument(fName);
  }
}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
std::auto_ptr<l1t::RegionalMuonCandBxCollection > OMTFReconstruction::reconstruct(const edm::Event& iEvent, const edm::EventSetup& evSetup) {

  loadAndFilterDigis(iEvent);

  if(dumpResultToXML) aTopElement = m_Writer->writeEventHeader(iEvent.id().event());

  // NOTE: assuming all is for bx 0
  int bx = 0;
  std::auto_ptr<l1t::RegionalMuonCandBxCollection > candidates(new l1t::RegionalMuonCandBxCollection);

  ///The order is important: first put omtf_pos candidates, then omtf_neg.
  for(unsigned int iProcessor=0; iProcessor<m_OMTFConfig->nProcessors(); ++iProcessor)
    getProcessorCandidates(iProcessor, l1t::tftype::omtf_pos, bx, *candidates);

  for(unsigned int iProcessor=0; iProcessor<m_OMTFConfig->nProcessors(); ++iProcessor)
    getProcessorCandidates(iProcessor, l1t::tftype::omtf_neg, bx, *candidates);
    
    return candidates;
}

/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
void OMTFReconstruction::loadAndFilterDigis(const edm::Event& iEvent){

  // Filter digis by dropping digis from selected (by cfg.py) subsystems
  if(!m_Config.getParameter<bool>("dropDTPrimitives")){
    iEvent.getByLabel(m_Config.getParameter<edm::InputTag>("srcDTPh"),dtPhDigis);
    iEvent.getByLabel(m_Config.getParameter<edm::InputTag>("srcDTTh"),dtThDigis);
  }
  if(!m_Config.getParameter<bool>("dropRPCPrimitives")) iEvent.getByLabel(m_Config.getParameter<edm::InputTag>("srcRPC"),rpcDigis);
  if(!m_Config.getParameter<bool>("dropCSCPrimitives")) iEvent.getByLabel(m_Config.getParameter<edm::InputTag>("srcCSC"),cscDigis);
  
}

/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
void OMTFReconstruction::getProcessorCandidates(unsigned int iProcessor, l1t::tftype mtfType, int bx,
               l1t::RegionalMuonCandBxCollection & OTFCandidates){
  
  OMTFinput input = m_InputMaker.buildInputForProcessor(dtPhDigis.product(),
                dtThDigis.product(),
                cscDigis.product(),
                rpcDigis.product(),
                iProcessor, mtfType);
  
  const std::vector<OMTFProcessor::resultsMap> & results = m_OMTF->processInput(iProcessor,input);

  std::vector<AlgoMuon> algoCandidates;

  m_Sorter.sortRefHitResults(results, algoCandidates);  
  m_GhostBuster.select(algoCandidates); 
  m_Sorter.sortProcessorAndFillCandidates(iProcessor, mtfType, algoCandidates, OTFCandidates, bx);

  writeResultToXML(iProcessor, input, results);
}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
void OMTFReconstruction::writeResultToXML(unsigned int iProcessor, const OMTFinput &input, 
               const std::vector<OMTFProcessor::resultsMap> & results){

  //Write data to XML file
  if(dumpResultToXML){
    xercesc::DOMElement * aProcElement = m_Writer->writeEventData(aTopElement,iProcessor,input);
    for(unsigned int iRefHit=0;iRefHit<m_OMTFConfig->nTestRefHits();++iRefHit){
      ///Dump only regions, where a candidate was found
      AlgoMuon myCand = m_Sorter.sortRefHitResults(results[iRefHit],0);//charge=0 means ignore charge
      if(myCand.getPt()) {
        m_Writer->writeCandidateData(aProcElement,iRefHit,myCand);
        if(dumpDetailedResultToXML){
          for(auto & itKey: results[iRefHit])
            m_Writer->writeResultsData(aProcElement, iRefHit, itKey.first,itKey.second);
        }
      }
    }
  }
}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////

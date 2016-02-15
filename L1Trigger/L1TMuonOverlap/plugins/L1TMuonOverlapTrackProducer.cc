#include <iostream>
#include <strstream>
#include <vector>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/DataRecord/interface/L1TMuonOverlapParamsRcd.h"
#include "CondFormats/L1TObjects/interface/L1TMuonOverlapParams.h"

#include "L1Trigger/L1TMuonOverlap/plugins/L1TMuonOverlapTrackProducer.h"
#include "L1Trigger/L1TMuonOverlap/interface/OMTFProcessor.h"
#include "L1Trigger/L1TMuonOverlap/interface/OMTFinput.h"
#include "L1Trigger/L1TMuonOverlap/interface/OMTFConfiguration.h"
#include "L1Trigger/L1TMuonOverlap/interface/XMLConfigWriter.h"

#include "L1Trigger/RPCTrigger/interface/RPCConst.h"

L1TMuonOverlapTrackProducer::L1TMuonOverlapTrackProducer(const edm::ParameterSet& cfg)
  :theConfig(cfg), myOMTFConfig(0), myOMTF(0), aTopElement(0), myOMTFConfigMaker(0), myWriter(0) {

  produces<l1t::RegionalMuonCandBxCollection >("OMTF");

  inputTokenDTPh = consumes<L1MuDTChambPhContainer>(theConfig.getParameter<edm::InputTag>("srcDTPh"));
  inputTokenDTTh = consumes<L1MuDTChambThContainer>(theConfig.getParameter<edm::InputTag>("srcDTTh"));
  inputTokenCSC = consumes<CSCCorrelatedLCTDigiCollection>(theConfig.getParameter<edm::InputTag>("srcCSC"));
  inputTokenRPC = consumes<RPCDigiCollection>(theConfig.getParameter<edm::InputTag>("srcRPC"));

  if(!theConfig.exists("omtf")){
    edm::LogError("L1TMuonOverlapTrackProducer")<<"omtf configuration not found in cfg.py";
  }

  dumpResultToXML = theConfig.getParameter<bool>("dumpResultToXML");
  dumpDetailedResultToXML = theConfig.getParameter<bool>("dumpDetailedResultToXML");
  theConfig.getParameter<std::string>("XMLDumpFileName");

  if(dumpResultToXML){
    myWriter = new XMLConfigWriter();
    std::string fName = "OMTF";
    myWriter->initialiseXMLDocument(fName);
  }
}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
L1TMuonOverlapTrackProducer::~L1TMuonOverlapTrackProducer(){

  delete myOMTFConfig;
  delete myOMTF;

  if (myWriter) delete myWriter;

}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
void L1TMuonOverlapTrackProducer::beginJob(){

  if(theConfig.exists("omtf")){
    myOMTFConfig = new OMTFConfiguration(theConfig.getParameter<edm::ParameterSet>("omtf"));
    myOMTF = new OMTFProcessor(theConfig.getParameter<edm::ParameterSet>("omtf"));
  }
}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
void L1TMuonOverlapTrackProducer::endJob(){

  if(dumpResultToXML){
    std::string fName = theConfig.getParameter<std::string>("XMLDumpFileName");
    myWriter->finaliseXMLDocument(fName);
  }  
}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
void L1TMuonOverlapTrackProducer::beginRun(edm::Run const& run, edm::EventSetup const& iSetup){

  ///If configuration is read from XML do not look at the DB.
  if(theConfig.getParameter<edm::ParameterSet>("omtf").getParameter<bool>("configFromXML")) return;  

  const L1TMuonOverlapParamsRcd& omtfParamsRcd = iSetup.get<L1TMuonOverlapParamsRcd>();
  
  edm::ESHandle<L1TMuonOverlapParams> omtfParamsHandle;
  omtfParamsRcd.get(omtfParamsHandle);

  const L1TMuonOverlapParams* omtfParams = omtfParamsHandle.product();
  if (!omtfParams) {
    edm::LogError("L1TMuonOverlapTrackProducer") << "Could not retrieve parameters from Event Setup" << std::endl;
  }

  myOMTFConfig->configure(omtfParams);
  myOMTF->configure(omtfParams);
  
}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
void L1TMuonOverlapTrackProducer::produce(edm::Event& iEvent, const edm::EventSetup& evSetup){

  std::ostringstream myStr;

  myInputMaker.initialize(evSetup); //FIXME shoun't it be in beginRun?
 
  loadAndFilterDigis(iEvent);

  if(dumpResultToXML) aTopElement = myWriter->writeEventHeader(iEvent.id().event());

  // NOTE: assuming all is for bx 0
  int bx = 0;

  std::auto_ptr<l1t::RegionalMuonCandBxCollection > myCandidates(new l1t::RegionalMuonCandBxCollection);
  ///The order is iportant: first put omtf_pos candidates, then omtf_neg.
  for(unsigned int iProcessor=0;iProcessor<OMTFConfiguration::nProcessors;++iProcessor) getProcessorCandidates(iProcessor, l1t::tftype::omtf_pos, bx, *myCandidates);
  for(unsigned int iProcessor=0;iProcessor<OMTFConfiguration::nProcessors;++iProcessor) getProcessorCandidates(iProcessor, l1t::tftype::omtf_neg, bx, *myCandidates);
      
  myStr<<" Number of candidates: "<<myCandidates->size(bx);
  edm::LogInfo("OMTFOMTFProducer")<<myStr.str();
  
  iEvent.put(myCandidates, "OMTF");
}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
void L1TMuonOverlapTrackProducer::loadAndFilterDigis(edm::Event& iEvent){

  ///Filter digis by dropping digis from selected (by cfg.py) subsystems
  if(!theConfig.getParameter<bool>("dropDTPrimitives")){
    iEvent.getByToken(inputTokenDTPh,dtPhDigis);
    iEvent.getByToken(inputTokenDTTh,dtThDigis);
  }
  if(!theConfig.getParameter<bool>("dropRPCPrimitives")) iEvent.getByToken(inputTokenRPC,rpcDigis);  
  if(!theConfig.getParameter<bool>("dropCSCPrimitives")) iEvent.getByToken(inputTokenCSC,cscDigis);
  
}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
void L1TMuonOverlapTrackProducer::getProcessorCandidates(unsigned int iProcessor, l1t::tftype mtfType, int bx,
							 l1t::RegionalMuonCandBxCollection & myOTFCandidates){
  
  OMTFinput myInput = myInputMaker.buildInputForProcessor(dtPhDigis.product(),
							  dtThDigis.product(),
							  cscDigis.product(),
							  rpcDigis.product(),
							  iProcessor, mtfType);
  
  const std::vector<OMTFProcessor::resultsMap> & myResults = myOMTF->processInput(iProcessor,myInput);
  mySorter.sortProcessorAndFillCandidates(iProcessor, mtfType, myResults, myOTFCandidates, bx);
  
  writeResultToXML(iProcessor, myInput, myResults);

}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
void L1TMuonOverlapTrackProducer::writeResultToXML(unsigned int iProcessor, const OMTFinput &myInput, 
						   const std::vector<OMTFProcessor::resultsMap> & myResults){

  ///Write data to XML file
  if(dumpResultToXML){
    xercesc::DOMElement * aProcElement = myWriter->writeEventData(aTopElement,iProcessor,myInput);
    for(unsigned int iRefHit=0;iRefHit<OMTFConfiguration::nTestRefHits;++iRefHit){
      ///Dump only regions, where a candidate was found
      InternalObj myCand = mySorter.sortRefHitResults(myResults[iRefHit],0);//charge=0 means ignore charge
      if(myCand.pt){
	myWriter->writeCandidateData(aProcElement,iRefHit,myCand);
	if(dumpDetailedResultToXML){
	  for(auto & itKey: myResults[iRefHit]) myWriter->writeResultsData(aProcElement,
									   iRefHit,
									   itKey.first,itKey.second);
	}
      }
    }
  }
}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1TMuonOverlapTrackProducer);


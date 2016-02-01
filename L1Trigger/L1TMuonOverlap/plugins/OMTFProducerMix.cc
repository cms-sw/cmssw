#include <iostream>
#include <strstream>
#include <vector>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"

#include "CondFormats/DataRecord/interface/L1TMuonOverlapParamsRcd.h"
#include "CondFormats/L1TObjects/interface/L1TMuonOverlapParams.h"

#include "L1Trigger/L1TMuonOverlap/plugins/OMTFProducerMix.h"
#include "L1Trigger/L1TMuonOverlap/interface/OMTFProcessor.h"
#include "L1Trigger/L1TMuonOverlap/interface/OMTFinputMaker.h"
#include "L1Trigger/L1TMuonOverlap/interface/OMTFinput.h"
#include "L1Trigger/L1TMuonOverlap/interface/OMTFSorter.h"
#include "L1Trigger/L1TMuonOverlap/interface/OMTFConfiguration.h"
#include "L1Trigger/L1TMuonOverlap/interface/OMTFConfigMaker.h"
#include "L1Trigger/L1TMuonOverlap/interface/XMLConfigWriter.h"
#include "L1Trigger/L1TMuonOverlap/interface/XMLConfigReader.h"

OMTFProducerMix::OMTFProducerMix(const edm::ParameterSet& cfg):
  theConfig(cfg){

  produces<l1t::RegionalMuonCandBxCollection >("OMTF");

  inputTokenDTPh = consumes<L1MuDTChambPhContainer>(theConfig.getParameter<edm::InputTag>("srcDTPh"));
  inputTokenDTTh = consumes<L1MuDTChambThContainer>(theConfig.getParameter<edm::InputTag>("srcDTTh"));
  inputTokenCSC = consumes<CSCCorrelatedLCTDigiCollection>(theConfig.getParameter<edm::InputTag>("srcCSC"));
  inputTokenRPC = consumes<RPCDigiCollection>(theConfig.getParameter<edm::InputTag>("srcRPC"));
  
  dumpResultToXML = theConfig.getParameter<bool>("dumpResultToXML");

  if(!theConfig.exists("omtf")){
    edm::LogError("OMTFProducerMix")<<"omtf configuration not found in cfg.py";
  }

  myInputMaker = new OMTFinputMaker();
  mySorter = new OMTFSorter();
  myWriter = 0;
  myReader = 0;

  myInputXML = new OMTFinput();
  myReader = new XMLConfigReader();
  if(dumpResultToXML){
    myWriter = new XMLConfigWriter();
    std::string fName = "OMTF_Events";
    myWriter->initialiseXMLDocument(fName);
  }

  std::vector<std::string> fileNames = theConfig.getParameter<std::vector<std::string> >("eventsXMLFiles");
  for(auto it: fileNames) myReader->setEventsFile(it);
  eventsToMix = theConfig.getParameter<unsigned int>("eventsToMix");

  myOMTFConfig = 0;
  myEventNumber = 0;
}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
OMTFProducerMix::~OMTFProducerMix(){

  delete myOMTFConfig;
  delete myOMTFConfigMaker;
  delete myOMTF;

  delete myInputMaker;
  delete mySorter;

  if(myWriter) delete myWriter;
  delete myReader;
  delete myInputXML;
}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
void OMTFProducerMix::beginRun(edm::Run const& run, edm::EventSetup const& iSetup){

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
void OMTFProducerMix::beginJob(){

  if(theConfig.exists("omtf")){
    myOMTFConfig = new OMTFConfiguration(theConfig.getParameter<edm::ParameterSet>("omtf"));
    myOMTFConfigMaker = new OMTFConfigMaker(theConfig.getParameter<edm::ParameterSet>("omtf"));
    myOMTF = new OMTFProcessor(theConfig.getParameter<edm::ParameterSet>("omtf"));
  }
}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
void OMTFProducerMix::endJob(){

  if(dumpResultToXML){
    std::string fName = "MixedEvents.xml";
    myWriter->finaliseXMLDocument(fName);
  }
}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
void OMTFProducerMix::produce(edm::Event& iEvent, const edm::EventSetup& evSetup){

  ++myEventNumber;
  unsigned int eventToSave = 252;

  myInputMaker->initialize(evSetup);

  edm::Handle<L1MuDTChambPhContainer> dtPhDigis;
  edm::Handle<L1MuDTChambThContainer> dtThDigis;
  edm::Handle<CSCCorrelatedLCTDigiCollection> cscDigis;
  edm::Handle<RPCDigiCollection> rpcDigis;
  
  ///Filter digis by dropping digis from selected (by cfg.py) subsystems
  if(!theConfig.getParameter<bool>("dropDTPrimitives")){
    iEvent.getByToken(inputTokenDTPh,dtPhDigis);
    iEvent.getByToken(inputTokenDTTh,dtThDigis);
  }
  if(!theConfig.getParameter<bool>("dropRPCPrimitives")) iEvent.getByToken(inputTokenRPC,rpcDigis);  
  if(!theConfig.getParameter<bool>("dropCSCPrimitives")) iEvent.getByToken(inputTokenCSC,cscDigis);

  std::auto_ptr<l1t::RegionalMuonCandBxCollection > myCands(new l1t::RegionalMuonCandBxCollection);

  // NOTE: for now just assuming it's central BX only:
  int bx = 0;
  ///Loop over events to be mixed with current EDM event
  for(unsigned int iEventMix=0;iEventMix<=2*eventsToMix;++iEventMix){
    edm::LogInfo("OMTFOMTFProducerMix")<<"iMix: "<<iEventMix;
    if(dumpResultToXML && myEventNumber==eventToSave && iEventMix==4) aTopElement = myWriter->writeEventHeader(iEvent.id().event(), iEventMix);

    ///Loop over all processors, each covering 60 deg in phi
    for(unsigned int iProcessor=0;iProcessor<6;++iProcessor){

      edm::LogInfo("OMTFOMTFProducerMix")<<" iProcessor: "<<iProcessor;
      OMTFinput myInput = myInputMaker->buildInputForProcessor(dtPhDigis.product(),
									 dtThDigis.product(),
									 cscDigis.product(),
									 rpcDigis.product(),								       
									 iProcessor,
									 l1t::tftype::omtf_pos);
      
      ///Every second BX contains the mixed event
      if(iEventMix%2==1 && iEventMix>0) myInput.clear();
      ///First BX contains the original event
      if(iEventMix>0){
	myInputXML->clear();
	myInputXML->readData(myReader,int(iEventMix-0.5)/2, iProcessor);
	myInput.mergeData(myInputXML);
      }
      ///Results for each GP in each logic region of given processor
      const std::vector<OMTFProcessor::resultsMap> & myResults = myOMTF->processInput(iProcessor,myInput);

      //Retreive all candidates returned by sorter: upto 3 non empty ones with different phi or charge
      l1t::RegionalMuonCandBxCollection  myOTFCandidates;
      mySorter->sortProcessor(myResults, myOTFCandidates, bx);

      ////Switch from internal processor n bit scale to global one
      int procOffset = OMTFConfiguration::globalPhiStart(iProcessor);
      int lowScaleEnd = pow(2,OMTFConfiguration::nPhiBits-1);
      if(procOffset<0) procOffset+=OMTFConfiguration::nPhiBins;


      for(unsigned int iCand=0; iCand<myOTFCandidates.size(bx); ++iCand){
	// shift phi from processor to global coordinates
        l1t::RegionalMuonCand cand = myOTFCandidates.at(bx, iCand);
	int phiValue = (cand.hwPhi()+procOffset+lowScaleEnd);
	if(phiValue>=(int)OMTFConfiguration::nPhiBins) phiValue-=OMTFConfiguration::nPhiBins;
	phiValue/=10; //MicroGMT has 10x coarser scale than OMTF
	cand.setHwPhi(phiValue);
	cand.setHwSignValid(iEventMix);
	// store candidate
	if(cand.hwPt()) myCands->push_back(bx, cand);
      }

      edm::LogInfo("OMTFOMTFProducerMix")<<" Number of candidates: "<<myOTFCandidates.size(bx);

      ///Write to XML
      if(dumpResultToXML && myEventNumber==eventToSave && iEventMix==4){
	xercesc::DOMElement * aProcElement = myWriter->writeEventData(aTopElement,iProcessor,myInput);
	for(unsigned int iRefHit=0;iRefHit<OMTFConfiguration::nTestRefHits;++iRefHit){
	  ///Dump only regions, where a candidate was found
	  InternalObj myCand = mySorter->sortRefHitResults(myResults[iRefHit],0);//charge=0 means ignore charge
	  if(myCand.pt){
	    myWriter->writeCandidateData(aProcElement,iRefHit,myCand);
	  }
	}
      }
    }
  }

  edm::LogInfo("OMTFOMTFProducerMix")<<" Number of candidates: "<<myCands->size(bx);

  iEvent.put(myCands, "OMTF");
}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(OMTFProducerMix);


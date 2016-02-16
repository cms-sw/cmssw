#include <iostream>
#include <strstream>
#include <vector>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"

#include "CondFormats/DataRecord/interface/L1TMuonOverlapParamsRcd.h"
#include "CondFormats/L1TObjects/interface/L1TMuonOverlapParams.h"

#include "L1Trigger/L1TMuonOverlap/plugins/L1TMuonOverlapTrackProducer.h"
#include "L1Trigger/L1TMuonOverlap/interface/OMTFProcessor.h"
#include "L1Trigger/L1TMuonOverlap/interface/OMTFinputMaker.h"
#include "L1Trigger/L1TMuonOverlap/interface/OMTFinput.h"
#include "L1Trigger/L1TMuonOverlap/interface/OMTFSorter.h"
#include "L1Trigger/L1TMuonOverlap/interface/OMTFConfiguration.h"
#include "L1Trigger/L1TMuonOverlap/interface/XMLConfigWriter.h"

#include "FWCore/Framework/interface/ESHandle.h"

L1TMuonOverlapTrackProducer::L1TMuonOverlapTrackProducer(const edm::ParameterSet& cfg):theConfig(cfg){

  produces<l1t::RegionalMuonCandBxCollection >("OMTF");

  inputTokenDTPh = consumes<L1MuDTChambPhContainer>(theConfig.getParameter<edm::InputTag>("srcDTPh"));
  inputTokenDTTh = consumes<L1MuDTChambThContainer>(theConfig.getParameter<edm::InputTag>("srcDTTh"));
  inputTokenCSC = consumes<CSCCorrelatedLCTDigiCollection>(theConfig.getParameter<edm::InputTag>("srcCSC"));
  inputTokenRPC = consumes<RPCDigiCollection>(theConfig.getParameter<edm::InputTag>("srcRPC"));

  if(!theConfig.exists("omtf")){
    edm::LogError("L1TMuonOverlapTrackProducer")<<"omtf configuration not found in cfg.py";
  }

  myInputMaker = new OMTFinputMaker();
  mySorter = new OMTFSorter();
  myOMTF = 0;
  myWriter = 0;

  dumpResultToXML = theConfig.getParameter<bool>("dumpResultToXML");
  dumpDetailedResultToXML = theConfig.getParameter<bool>("dumpDetailedResultToXML");
  dumpGPToXML = theConfig.getParameter<bool>("dumpGPToXML");
  theConfig.getParameter<std::string>("XMLDumpFileName");

  if(dumpResultToXML || dumpGPToXML){
    myWriter = new XMLConfigWriter();
    std::string fName = "OMTF";
    myWriter->initialiseXMLDocument(fName);
  }

  myOMTFConfig = 0;
}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
L1TMuonOverlapTrackProducer::~L1TMuonOverlapTrackProducer(){

  delete myOMTFConfig;
  delete myOMTF;

  delete myInputMaker;
  delete mySorter;

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

  if(dumpResultToXML && !dumpGPToXML){
    std::string fName = theConfig.getParameter<std::string>("XMLDumpFileName");
    myWriter->finaliseXMLDocument(fName);
  }

  if(dumpGPToXML && !dumpResultToXML){

    GoldenPattern *dummy = new GoldenPattern(Key(0,0,0));
    dummy->reset();

    std::string fName = "OMTF";
    myWriter->initialiseXMLDocument(fName);
    const std::map<Key,GoldenPattern*> & myGPmap = myOMTF->getPatterns();
    for(auto itGP: myGPmap){
      if(itGP.second->key().thePtCode==6) std::cout<<*itGP.second<<std::endl;
      //myWriter->writeGPData(*itGP.second);
      if(itGP.second->key().thePtCode>5) myWriter->writeGPData(*itGP.second,*dummy, *dummy, *dummy);
    }
    fName = "GPs.xml";
    myWriter->finaliseXMLDocument(fName);
    ///Write GPs merged by 4 above iPt19, and by 2 below//
    //////////////////////////////////////////////////////
    ///4x merging
    fName = "OMTF";
    myWriter->initialiseXMLDocument(fName);
    myOMTF->averagePatterns(1);
    myOMTF->averagePatterns(-1);

    writeMergedGPs();

    fName = "GPs_4x.xml";
    myWriter->finaliseXMLDocument(fName);
  }
}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
void L1TMuonOverlapTrackProducer::writeMergedGPs(){
  
  const std::map<Key,GoldenPattern*> & myGPmap = myOMTF->getPatterns();

  GoldenPattern *dummy = new GoldenPattern(Key(0,0,0));
  dummy->reset();

  unsigned int iPtMin = 6;
  Key aKey = Key(1, iPtMin,-1);
  while(myGPmap.find(aKey)!=myGPmap.end()){

    GoldenPattern *aGP1 = myGPmap.find(aKey)->second;
    GoldenPattern *aGP2 = dummy;
    GoldenPattern *aGP3 = dummy;
    GoldenPattern *aGP4 = dummy;

    ++aKey.thePtCode;
    if(aKey.thePtCode<=31 && myGPmap.find(aKey)!=myGPmap.end()) aGP2 =  myGPmap.find(aKey)->second;

    if(aKey.thePtCode>19){
      ++aKey.thePtCode;
      if(aKey.thePtCode<=31 && myGPmap.find(aKey)!=myGPmap.end()) aGP3 =  myGPmap.find(aKey)->second;

      ++aKey.thePtCode;
      if(aKey.thePtCode<=31 && myGPmap.find(aKey)!=myGPmap.end()) aGP4 =  myGPmap.find(aKey)->second;
    }
    ++aKey.thePtCode;
    myWriter->writeGPData(*aGP1,*aGP2, *aGP3, *aGP4);

    ///Write the opposite charge.
    Key aTmpKey = aGP1->key();
    aTmpKey.theCharge = 1;
    if(myGPmap.find(aTmpKey)!=myGPmap.end()) aGP1 =  myGPmap.find(aTmpKey)->second;
    else aGP1 = dummy;

    aTmpKey = aGP2->key();
    aTmpKey.theCharge = 1;
    if(myGPmap.find(aTmpKey)!=myGPmap.end()) aGP2 =  myGPmap.find(aTmpKey)->second;
    else aGP2 = dummy;

    aTmpKey = aGP3->key();
    aTmpKey.theCharge = 1;
    if(myGPmap.find(aTmpKey)!=myGPmap.end()) aGP3 =  myGPmap.find(aTmpKey)->second;
    else aGP3 = dummy;

    aTmpKey = aGP4->key();
    aTmpKey.theCharge = 1;
    if(myGPmap.find(aTmpKey)!=myGPmap.end()) aGP4 =  myGPmap.find(aTmpKey)->second;
    else aGP4 = dummy;
    
    myWriter->writeGPData(*aGP1,*aGP2, *aGP3, *aGP4);
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

  omtfParams = std::unique_ptr<L1TMuonOverlapParams>(new L1TMuonOverlapParams(*omtfParamsHandle.product()));
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

  if(dumpResultToXML) aTopElement = myWriter->writeEventHeader(iEvent.id().event());

  // NOTE: assuming all is for bx 0
  int bx = 0;

  ///Loop over all processors, each covering 60 deg in phi
  for(unsigned int iProcessor=0;iProcessor<6;++iProcessor){

    //myStr<<" iProcessor: "<<iProcessor;

    ///Input data with phi ranges shifted for each processor, so it fits 11 bits range
    const OMTFinput *myInputPos = myInputMaker->buildInputForProcessor(dtPhDigis.product(),
								       dtThDigis.product(),
								       cscDigis.product(),
								       rpcDigis.product(),								       
								       iProcessor,
								       l1t::tftype::omtf_pos);

    OMTFinput myShiftedInputPos =  myOMTF->shiftInput(iProcessor,*myInputPos);

    const OMTFinput *myInputNeg = myInputMaker->buildInputForProcessor(dtPhDigis.product(),
								       dtThDigis.product(),
								       cscDigis.product(),
								       rpcDigis.product(),								       
								       iProcessor,
								       l1t::tftype::omtf_neg);

    
    OMTFinput myShiftedInputNeg =  myOMTF->shiftInput(iProcessor,*myInputNeg);

    l1t::RegionalMuonCandBxCollection myOTFCandidatesPos, myOTFCandidatesNeg;
    ///Results for each GP in each logic region of given processor
    //Retreive all candidates returned by sorter: upto 3 non empty ones with different phi or charge
    const std::vector<OMTFProcessor::resultsMap> & myResultsNeg = myOMTF->processInput(iProcessor,myShiftedInputNeg);
    mySorter->sortProcessor(myResultsNeg, myOTFCandidatesNeg, bx);

    const std::vector<OMTFProcessor::resultsMap> & myResultsPos = myOMTF->processInput(iProcessor,myShiftedInputPos);
    mySorter->sortProcessor(myResultsPos, myOTFCandidatesPos, bx);

    ///Shift phi scales, and put MicroGMT candidates into myCands collection
    processCandidates(iProcessor, bx, myCands, myOTFCandidatesPos, l1t::tftype::omtf_pos);
    processCandidates(iProcessor, bx, myCands, myOTFCandidatesNeg, l1t::tftype::omtf_neg);

    ///Write data to XML file
    if(dumpResultToXML){
      xercesc::DOMElement * aProcElement = myWriter->writeEventData(aTopElement,iProcessor,myShiftedInputPos);
      for(unsigned int iRefHit=0;iRefHit<OMTFConfiguration::nTestRefHits;++iRefHit){
	///Dump only regions, where a candidate was found
	InternalObj myCand = mySorter->sortRefHitResults(myResultsPos[iRefHit],0);//charge=0 means ignore charge
	if(myCand.pt){
	  myWriter->writeCandidateData(aProcElement,iRefHit,myCand);
	  if(dumpDetailedResultToXML){
	    for(auto & itKey: myResultsNeg[iRefHit]) myWriter->writeResultsData(aProcElement,
										iRefHit,
										itKey.first,itKey.second);
	  }
	}
      }
    }
  }

  //dumpResultToXML = true;
  myStr<<" Number of candidates: "<<myCands->size(bx);
  edm::LogInfo("OMTFOMTFProducer")<<myStr.str();

  iEvent.put(myCands, "OMTF");
}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
void L1TMuonOverlapTrackProducer::processCandidates(unsigned int iProcessor, int bx,
				     std::auto_ptr<l1t::RegionalMuonCandBxCollection > & myCands,
				     l1t::RegionalMuonCandBxCollection & myOTFCandidates,
				     l1t::tftype mtfType){

  ////Switch from internal processor n bit scale to global one
  int procOffset = OMTFConfiguration::globalPhiStart(iProcessor);
  if(procOffset<0) procOffset+=(int)OMTFConfiguration::nPhiBins;
  ///Set local 0 at iProcessor x 15 deg
  procOffset-=(15+iProcessor*60)/360.0*OMTFConfiguration::nPhiBins;
  int lowScaleEnd = pow(2,OMTFConfiguration::nPhiBits-1);

    for(unsigned int iCand=0; iCand<myOTFCandidates.size(bx); ++iCand){
      // shift phi from processor to global coordinates
      l1t::RegionalMuonCand cand = myOTFCandidates.at(bx, iCand);
      int phiValue = (cand.hwPhi()+procOffset+lowScaleEnd);
      if(phiValue>=(int)OMTFConfiguration::nPhiBins) phiValue-=OMTFConfiguration::nPhiBins;
      phiValue/=10; //MicroGMT has 10x coarser scale than OMTF

      cand.setHwPhi(phiValue);
      cand.setTFIdentifiers(iProcessor,mtfType);
      // store candidate
      if(cand.hwPt()) myCands->push_back(bx, cand);
    }
}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1TMuonOverlapTrackProducer);

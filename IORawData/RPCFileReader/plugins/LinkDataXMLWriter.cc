#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "EventFilter/RPCRawToDigi/interface/RPCPackingModule.h"
#include "CondFormats/RPCObjects/interface/RPCReadOutMapping.h"
#include "CondFormats/DataRecord/interface/RPCReadOutMappingRcd.h"
#include "EventFilter/RPCRawToDigi/interface/EventRecords.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "EventFilter/RPCRawToDigi/interface/RPCRecordFormatter.h"

#include "IORawData/RPCFileReader/interface/LinkDataXMLWriter.h"

#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/framework/LocalFileFormatTarget.hpp>
#include "boost/bind.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>

XERCES_CPP_NAMESPACE_USE

using namespace std;
using namespace edm;
using namespace rpcrawtodigi;
/*############################################################################
#
#  Needed for xerces to work
#
############################################################################*/
class XStr
{
public :
    XStr(const char* const toTranscode)
    {
        fUnicodeForm = XMLString::transcode(toTranscode);
    }

    ~XStr()
    {
        XMLString::release(&fUnicodeForm);
    }

    const XMLCh* unicodeForm() const
    {
        return fUnicodeForm;
    }

private :
    XMLCh*   fUnicodeForm;
};

#define X(str) XStr(str).unicodeForm()
/*############################################################################*/
int LinkDataXMLWriter::m_instanceCount = 0;
/*############################################################################
#
# Constructor
#
############################################################################*/
LinkDataXMLWriter::LinkDataXMLWriter(const edm::ParameterSet& iConfig){


   digiLabel = iConfig.getParameter<edm::InputTag>("digisSource");

   m_xmlDir = iConfig.getParameter<std::string>("xmlDir");

  if (m_instanceCount == 0) {
    try {
        XMLPlatformUtils::Initialize();
        ++m_instanceCount;
    }
    catch(const XMLException &toCatch)  {
        throw cms::Exception("xmlError") << ("Error during Xerces-c Initialization: "
           + std::string(XMLString::transcode(toCatch.getMessage())));
    }
  }

  ///Set root element.
  DOMImplementation* impl =  DOMImplementationRegistry::getDOMImplementation(X("Core"));

  if (impl != NULL) {

    doc = impl->createDocument(0,                    // root element namespace URI.
			       X("rpctDataStream"),         // root element name
			       0);                   // document type object (DTD).
    
    rootElem = doc->getDocumentElement();
  }
    else {
      throw cms::Exception("xmlError") << "Could'n get DOMImplementation\n";
    }

  ///Setup output
  XMLCh tempStr[100];
  XMLString::transcode("LS", tempStr, 99);
  DOMImplementation *impl_1          = DOMImplementationRegistry::getDOMImplementation(tempStr);
  theSerializer = ((DOMImplementationLS*)impl_1)->createDOMWriter();
  
  // set user specified output encoding
  theSerializer->setEncoding(X("UTF-8"));
  
  if (theSerializer->canSetFeature(XMLUni::fgDOMWRTFormatPrettyPrint, true))
    theSerializer->setFeature(XMLUni::fgDOMWRTFormatPrettyPrint, true);
  
  std::string outFileName = m_xmlDir;
  myFormTarget = new LocalFileFormatTarget(X(outFileName.c_str()));
  
  DOMNode* xmlstylesheet  = doc->createProcessingInstruction(X("xml-stylesheet"),
							     X("type=\"text/xsl\"href=\"default.xsl\""));

  doc->insertBefore(xmlstylesheet, rootElem);
  //////////////////////////////////////////////
  event = doc->createElement(X("event"));
  event->setAttribute(X("bx"), X( IntToString(0).c_str()));
  event->setAttribute(X("num"), X( IntToString(0).c_str()));
  rootElem->appendChild(event);
  

  //set vector sizes
  nTC = 12;
  nTB = 9;
  for(int iTC=0;iTC<nTC;iTC++){
    std::vector< RPCPacData> rpdv(18,  RPCPacData());
    std::vector<std::vector< RPCPacData> > rpdvv(18,rpdv); 
    std::vector<std::vector<std::vector< RPCPacData> > > rpdvvv(nTB,rpdvv); 
    linkData.push_back(rpdvvv);
  }

  nEvents = 0; 
}

/*############################################################################
#
#
############################################################################*/
LinkDataXMLWriter::~LinkDataXMLWriter(){

  theSerializer->writeNode(myFormTarget, *doc);
  doc->release();

  delete theSerializer;
  delete myFormTarget;
  


}
/*############################################################################
#
#
############################################################################*/
void LinkDataXMLWriter::analyze(const edm::Event& ev, const edm::EventSetup& es){

  /// Get Data from all FEDs
  Handle< RPCDigiCollection > digiCollection;
  ev.getByLabel(digiLabel, digiCollection);
  /////////////// Print RPC digis
  RPCDigiCollection::DigiRangeIterator rpcDigiCI;
  for(rpcDigiCI = digiCollection->begin();rpcDigiCI!=digiCollection->end();rpcDigiCI++){
    cout<<(*rpcDigiCI).first<<endl;
    RPCDetId detId=(*rpcDigiCI).first;
    uint32_t id=detId();

    const RPCDigiCollection::Range& range = (*rpcDigiCI).second;
    for (RPCDigiCollection::const_iterator digiIt = range.first;
         digiIt!=range.second;++digiIt) cout<<"Digi: "<<*digiIt<<endl;
  }
 
 
  ESHandle<RPCReadOutMapping> readoutMapping;
  es.get<RPCReadOutMappingRcd>().get(readoutMapping);

  int trigger_BX = 200;
  int dccFactor = -1;
  
   pair<int,int> rpcFEDS=FEDNumbering::getRPCFEDIds();
   for (int id= rpcFEDS.first; id<=rpcFEDS.second; ++id){
     dccFactor++;

    RPCRecordFormatter formatter(id, readoutMapping.product()) ;
    std::vector<rpcrawtodigi::EventRecords> myEventRecords = RPCPackingModule::eventRecords(id,
											    trigger_BX,  
											    digiCollection.product(),
											    formatter); 
    std::cout<<" FED id: "<< id;
    //std::cout<<" myEventRecords.size(): "<< myEventRecords.size()<<std::endl;
    
    std::vector<rpcrawtodigi::EventRecords>::const_iterator CI =  myEventRecords.begin();
    for(;CI!= myEventRecords.end();CI++){ 

      int dccInputChannelNum =  CI->tbRecord().rmb();
      int opticalLinkNum =   CI->tbRecord().tbLinkInputNumber();
      int partitionData =   CI->lbRecord().lbData().lbData(); 
      int halfP = CI->lbRecord().lbData().halfP(); 
      int eod = CI->lbRecord().lbData().eod(); 
      int partitionNumber = CI->lbRecord().lbData().partitionNumber();  
      int lbNumber = CI->lbRecord().lbData().lbNumber();

      std::pair<int,int> aPair = getTCandTBNumbers(dccInputChannelNum,dccFactor);
      int triggerCrateNum = aPair.first;
      int triggerBoardNum = aPair.second;
      int a = aPair.second;
    
      addLinkData(triggerCrateNum, triggerBoardNum, opticalLinkNum, 
		  lbNumber, partitionNumber,  partitionData, halfP, eod);

    }
   }
    

   /*
    int triggerCrateNum = 9;
    int halfP = 0;
    int eod = 0;
    for(int iTB=0;iTB<9;iTB++){
      int iOL = nEvents%18;
      //for(int iOL=0;iOL<18;iOL++){
      int iPartNum = nEvents%12;
      //for(int iPartNum=0;iPartNum<8;iPartNum++){	    
	for(int iLbNum=0;iLbNum<3;iLbNum++){
	  int partitionData = iPartNum*2+1;  
	  addLinkData(triggerCrateNum, iTB, iOL, iLbNum, 
		      iPartNum, partitionData, halfP, eod);	  
	}
	//}
	//}
    }
   */
    writeLinkData();  
    
}
/*############################################################################
#
#
############################################################################*/
void LinkDataXMLWriter::addLinkData(int triggerCrateNum, int  triggerBoardNum, 
				    int opticalLinkNum, int lbNumber, 
				    int partitionNumber,  int partitionData, 
				    int halfPart, int eod){

  int partDelay = 0;

  RPCPacData myLinkData( partitionData,  partitionNumber,
			 partDelay, eod, halfPart, lbNumber);
  /*
  std::cout<<" opticalLinkNum: "<<opticalLinkNum
	   <<" lbNumber: "<<lbNumber
	   <<" rawData: "<<std::hex<<myLinkData.toRaw()<<std::dec<<std::endl;
  */
  int delay = 2;
  for(delay=0;delay<15;delay++) {
    /*
    std::cout<<"delay: "<<delay<<" raw: "
	     <<std::hex
	     <<linkData[triggerCrateNum][triggerBoardNum][delay][opticalLinkNum-1].toRaw()
	     <<std::dec<<std::endl;
*/
 
    if(!linkData[triggerCrateNum][triggerBoardNum][delay][opticalLinkNum].toRaw()) break;
  }

  linkData[triggerCrateNum][triggerBoardNum][delay][opticalLinkNum] = myLinkData;
}
/*############################################################################
#
#
############################################################################*/
void LinkDataXMLWriter::writeLinkData(){

  int bxNum = nEvents*10; //Leave room for data multiplexing.
  nEvents++;

  time_t timer;
  struct tm *tblock;
  timer = time(NULL);
  tblock = localtime(&timer);

  DOMElement* bx = doc->createElement(X("bxData"));
  bx->setAttribute(X("num"), X( IntToString(bxNum).c_str()));
  event->appendChild(bx);

  DOMElement*  tc = 0;
  DOMElement*  tb = 0;
  DOMElement*  ol = 0;

  for(int triggerCrateNum=0;triggerCrateNum<nTC;triggerCrateNum++){
    ////////////////////////////////////////////
    bool nonEmpty = false;
    for(int triggerBoardNum=0;triggerBoardNum<nTB;triggerBoardNum++){
      for(int opticalLinkNum=0;opticalLinkNum<18;opticalLinkNum++){
	for(int iDelay=0;iDelay<18;iDelay++){
	  int rawData =  linkData[triggerCrateNum][triggerBoardNum][iDelay][opticalLinkNum].toRaw();
	  if(rawData) nonEmpty = true;
	}
      }
    }
    if(!nonEmpty) continue;
    ////////////////////////////////////////////
    tc = doc->createElement(X("tc"));
    tc->setAttribute(X("num"), X( IntToString(triggerCrateNum).c_str()));
    for(int triggerBoardNum=0;triggerBoardNum<nTB;triggerBoardNum++){
      ////////////////////////////////////////////
      bool nonEmpty = false;
      for(int opticalLinkNum=0;opticalLinkNum<18;opticalLinkNum++){
	for(int iDelay=0;iDelay<18;iDelay++){
	  int rawData =  linkData[triggerCrateNum][triggerBoardNum][iDelay][opticalLinkNum].toRaw();
	  if(rawData) nonEmpty = true;
	}
      }
      if(!nonEmpty) continue;
      ////////////////////////////////////////////
      tb = doc->createElement(X("tb"));
      tb->setAttribute(X("num"), X( IntToString(triggerBoardNum).c_str()));
      for(int opticalLinkNum=0;opticalLinkNum<18;opticalLinkNum++){
        if(!linkData[triggerCrateNum][triggerBoardNum][0][opticalLinkNum].toRaw()) continue;
	ol = doc->createElement(X("ol"));
	ol->setAttribute(X("num"), X( IntToString(opticalLinkNum).c_str()));
	for(int iDelay=0;iDelay<18;iDelay++){
	  int rawData =  linkData[triggerCrateNum][triggerBoardNum][iDelay][opticalLinkNum].toRaw();
	  if(!rawData) continue;
	  const RPCPacData & myLinkData = linkData[triggerCrateNum][triggerBoardNum][iDelay][opticalLinkNum];
	  DOMElement*  lmd = doc->createElement(X("lmd"));
	  lmd->setAttribute(X("lb"), X( IntToString(myLinkData.lbNum()).c_str()));
	  lmd->setAttribute(X("par"), X( IntToString(myLinkData.partitionNum()).c_str()));
	  lmd->setAttribute(X("dat"), X( IntToString(myLinkData.partitionData(),1).c_str()));
	  lmd->setAttribute(X("del"), X( IntToString(myLinkData.partitionDelay()).c_str()));
	  lmd->setAttribute(X("eod"), X( IntToString(myLinkData.endOfData()).c_str()));
	  lmd->setAttribute(X("hp"), X( IntToString(myLinkData.halfPartition()).c_str()));
	  lmd->setAttribute(X("raw"), X( IntToString(myLinkData.toRaw(),1).c_str()));
	  ol->appendChild(lmd);
	}
	tb->appendChild(ol);
      }
      tc->appendChild(tb);      
    }
    bx->appendChild(tc);
  }
  clear();
}

std::string LinkDataXMLWriter::IntToString(int i, int opt){

 std::stringstream ss;
 if(opt==1) ss << std::hex << i << std::dec;
 else ss << i ;

 return ss.str();

}

void  LinkDataXMLWriter::clear(){

  int nTC = 12;
  int nTB = 9;

  for(int triggerCrateNum=0;triggerCrateNum<nTC;triggerCrateNum++){
    for(int triggerBoardNum=0;triggerBoardNum<nTB;triggerBoardNum++){
      for(int opticalLinkNum=0;opticalLinkNum<18;opticalLinkNum++){
	for(int iDelay=0;iDelay<18;iDelay++){
	  linkData[triggerCrateNum][triggerBoardNum][iDelay][opticalLinkNum] = RPCPacData();
	}
      }
    }
  }

}


std::pair<int,int>  LinkDataXMLWriter::getTCandTBNumbers(int dccInputChannelNum,int dccFactor){

  int tcNumber = 0;
  int tbNumber = 0;

  for(int i=0;i<9;i++){
    if(dccInputChannelNum==i ||
       dccInputChannelNum==9+i ||
       dccInputChannelNum==18+i ||
       dccInputChannelNum==27+i) tbNumber = i;
  }
  /////////////////////
  cout<<"dcc: "<<dccInputChannelNum<<endl;
  /////////////////////
 for(int i=0;i<4;i++){
    if(dccInputChannelNum>=i*9 &&
       dccInputChannelNum<9+i*9) tcNumber = i; //Count TC from 0, not from 1.

  }

 tcNumber+=4*dccFactor;

  return std::pair<int,int>(tcNumber,tbNumber);

}

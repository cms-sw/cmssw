#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "IORawData/RPCFileReader/interface/LinkDataXMLWriter.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/framework/LocalFileFormatTarget.hpp>
#include "boost/bind.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>

XERCES_CPP_NAMESPACE_USE
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
//LinkDataXMLWriter::LinkDataXMLWriter(const edm::ParameterSet& iConfig){
LinkDataXMLWriter::LinkDataXMLWriter(){

  m_xmlDir = "/afs/cern.ch/user/a/akalinow/scratch0/MTCC_II/XMLLinkData/";

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


  //set vector sizes
  for(int iTC=0;iTC<12;iTC++){
    std::vector< RPCPacData> rpdv(17,  RPCPacData());
    std::vector<std::vector< RPCPacData> > rpdvv(18,rpdv); 
    std::vector<std::vector<std::vector< RPCPacData> > > rpdvvv(18,rpdvv); 
    linkData.push_back(rpdvvv);
  }
}

/*############################################################################
#
#
############################################################################*/
LinkDataXMLWriter::~LinkDataXMLWriter(){

  std::cout<<"~LinkDataXMLWriter"<<std::endl;

  writeLinkData();

  // save file
  XMLCh tempStr[100];
  XMLString::transcode("LS", tempStr, 99);
  DOMImplementation *impl          = DOMImplementationRegistry::getDOMImplementation(tempStr);
  DOMWriter         *theSerializer = ((DOMImplementationLS*)impl)->createDOMWriter();
  
  // set user specified output encoding
  theSerializer->setEncoding(X("UTF-8"));
  
  if (theSerializer->canSetFeature(XMLUni::fgDOMWRTFormatPrettyPrint, true))
    theSerializer->setFeature(XMLUni::fgDOMWRTFormatPrettyPrint, true);

  XMLFormatTarget *myFormTarget;

  std::string outFileName = m_xmlDir + "/" + "testBxData.xml";
  myFormTarget = new LocalFileFormatTarget(X(outFileName.c_str()));
  
  DOMNode* xmlstylesheet  = doc->createProcessingInstruction(X("xml-stylesheet"),
							     X("type=\"text/xsl\"href=\"default.xsl\""));

  doc->insertBefore(xmlstylesheet, rootElem);
  theSerializer->writeNode(myFormTarget, *doc);
  
  delete theSerializer;
  delete myFormTarget;
  
  doc->release();

}
/*############################################################################
#
#
############################################################################*/
/*
virtual void analyze(const edm::Event&, const edm::EventSetup&){

  /// Get Data from all FEDs
  Handle<FEDRawDataCollection> allFEDRawData; 
  e.getByType(allFEDRawData); 

  edm::ESHandle<RPCReadOutMapping> readoutMapping;
  c.get<RPCReadOutMappingRcd>().get(readoutMapping);

  std::auto_ptr<RPCDigiCollection> producedRPCDigis(new RPCDigiCollection);
  std::pair<int,int> rpcFEDS=FEDNumbering::getRPCFEDIds();
 
  for (int id= rpcFEDS.first; id<=rpcFEDS.second; ++id){  

    const FEDRawData & fedData = allFEDRawData->FEDData(id);
    RPCRecordFormatter interpreter(id, readoutMapping.product()) ;
    RPCFEDData rpcRawData;

    if(fedData.size()){
    const unsigned char* index = fedData.data();
    
    }
    }
    }
*/
/*############################################################################
#
#
############################################################################*/
void LinkDataXMLWriter::addLinkData(int triggerCrateNum, int triggerBoardNum, 
				    int opticalLinkNum, int lbNumber, 
				    int partitionNumber,  int partitionData, 
				    int halfPart, int eod){

  int partDelay = 0;

  RPCPacData myLinkData( partitionData,  partitionNumber,
			 partDelay, eod, halfPart, lbNumber);
  /*
  std::cout<<"opticalLinkNum: "<<opticalLinkNum
	   <<" lbNumber: "<<lbNumber<<std::endl;
  std::cout<<"rawData: "<<std::hex<<myLinkData.toRaw()<<std::dec<<std::endl;
  */
  int delay = 2;
  for(delay=0;delay<15;delay++) {
    //std::cout<<"delay: "<<delay<<" raw: "
//	     <<std::hex
//	     <<linkData[triggerCrateNum][triggerBoardNum][delay][opticalLinkNum-1].toRaw()
//	     <<std::dec<<std::endl;
    if(!linkData[triggerCrateNum][triggerBoardNum][delay][opticalLinkNum-1].toRaw()) break;
  }

  linkData[triggerCrateNum][triggerBoardNum][delay][opticalLinkNum-1] = myLinkData;
}
/*############################################################################
#
#
############################################################################*/
void LinkDataXMLWriter::writeLinkData(){

  int bxNum = 0;

  time_t timer;
  struct tm *tblock;
  timer = time(NULL);
  tblock = localtime(&timer);

  DOMElement* bx = doc->createElement(X("bxData"));
  bx->setAttribute(X("num"), X( IntToString(bxNum).c_str()));
  rootElem->appendChild(bx);

  DOMElement*  tc = 0;
  DOMElement*  tb = 0;
  DOMElement*  ol = 0;

  for(int triggerCrateNum=0;triggerCrateNum<12;triggerCrateNum++){
    tc = doc->createElement(X("tc"));
    tc->setAttribute(X("num"), X( IntToString(triggerCrateNum).c_str()));
    for(int triggerBoardNum=0;triggerBoardNum<12;triggerBoardNum++){
      tb = doc->createElement(X("tb"));
      tb->setAttribute(X("num"), X( IntToString(triggerBoardNum).c_str()));
      for(int opticalLinkNum=0;opticalLinkNum<17;opticalLinkNum++){
	ol = doc->createElement(X("ol"));
	ol->setAttribute(X("num"), X( IntToString(opticalLinkNum).c_str()));
	for(int iDelay=0;iDelay<18;iDelay++){
	  int rawData =  linkData[triggerCrateNum][triggerBoardNum][iDelay][opticalLinkNum].toRaw();
	  if(!rawData) continue;
	  const RPCPacData & myLinkData = linkData[triggerCrateNum][triggerBoardNum][iDelay][opticalLinkNum];
	  //std::cout<<"raw data: "<<std::hex<<rawData<<std::dec<<std::endl;
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

  //DOMRange* range = doc->createRange();
  //range->release();

}

std::string LinkDataXMLWriter::IntToString(int i, int opt){

 std::stringstream ss;
 if(opt==1) ss << std::hex << i << std::dec;
 else ss << i ;

 return ss.str();

}



#include "IORawData/RPCFileReader/interface/XMLDataIO.h"
//#include "CondCore/ESSources/interface/Exception.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "FWCore/Framework/interface/Event.h"
#include "IORawData/RPCFileReader/interface/OptoTBData.h"

#include "boost/bind.hpp"
#include <iostream>


XMLDataIO::XMLDataIO(const std::string &fileName)
{
  std::cout <<"INIT" << std::endl;
  try {
    XMLPlatformUtils::Initialize();
  }
  catch(const XMLException &toCatch)  {
        throw cms::Exception("xmlError") << ("Error during Xerces-c Initialization: "
           + std::string(XMLString::transcode(toCatch.getMessage())));
  }
  DOMImplementation* impl =  DOMImplementationRegistry::getDOMImplementation(X("Core"));
  if (!impl) throw cms::Exception("xmlError") << "Could'n get DOMImplementation\n";

  doc = impl->createDocument(
                      0,                           // root element namespace URI.
                      X("rpctDataStream"),         // root element name
                      0);                          // document type object (DTD).

  rootElem = doc->getDocumentElement();

  ///Setup output
  XMLCh tempStr[100];
  XMLString::transcode("LS", tempStr, 99);
  DOMImplementation *impl_1          = DOMImplementationRegistry::getDOMImplementation(tempStr);
  theSerializer = ((DOMImplementationLS*)impl_1)->createDOMWriter();

  // set user specified output encoding
  theSerializer->setEncoding(X("UTF-8"));

  if (theSerializer->canSetFeature(XMLUni::fgDOMWRTFormatPrettyPrint, true))
    theSerializer->setFeature(XMLUni::fgDOMWRTFormatPrettyPrint, true);

  myFormTarget = new LocalFileFormatTarget(X(fileName.c_str()));

  DOMNode* xmlstylesheet  = doc->createProcessingInstruction(X("xml-stylesheet"),
                                               X("type=\"text/xsl\"href=\"default.xsl\""));

  doc->insertBefore(xmlstylesheet, rootElem);
}

XMLDataIO::~XMLDataIO()
{
  theSerializer->writeNode(myFormTarget, *doc);
  doc->release();
  delete theSerializer;
  delete myFormTarget;
}

void XMLDataIO::write(const edm::Event& ev, const std::vector<OptoTBData> & optoData)
{
  DOMElement * event = doc->createElement(X("event"));
  event->setAttribute(X("bx"), X( IntToString(ev.bunchCrossing()).c_str()));
  event->setAttribute(X("num"), X( IntToString(ev.id().event()).c_str()));
  DOMElement * bx=0;
  DOMElement * tc=0;
  DOMElement * tb=0;
  DOMElement * ol=0;
  unsigned int lastBX = 9999;
  unsigned lastTC = 9999;
  unsigned lastTB = 9999;
  unsigned lastOL = 9999;
  for (std::vector<OptoTBData>::const_iterator it = optoData.begin(); it < optoData.end(); ++it) {
    if(lastBX != it->bx()) {
      if(ol) { tb->appendChild(ol); ol = 0; } 
      if(tb) { tc->appendChild(tb); tb = 0; } 
      if(tc) { bx->appendChild(tc); tc = 0; } 
      if(bx) { event->appendChild(bx); bx = 0;}
      lastBX = it->bx();
      lastTC = 9999;
      lastTB = 9999;
      lastOL = 9999;
      bx = doc->createElement(X("bxData"));
      bx->setAttribute(X("num"), X( IntToString(lastBX).c_str()));
    }
    if (lastTC != it->tc()) {
      if(ol) { tb->appendChild(ol); ol = 0; }
      if(tb) { tc->appendChild(tb); tb = 0; }
      if(tc) { bx->appendChild(tc); tc = 0; }
      lastTC = it->tc();
      lastTB = 9999;
      lastOL = 9999;
      tc = doc->createElement(X("tc"));
      tc->setAttribute(X("num"), X( IntToString(lastTC).c_str()));
    }
    if (lastTB != it->tb()) {
      if(ol) { tb->appendChild(ol); ol = 0; }
      if(tb) { tc->appendChild(tb); tb = 0; }
      lastTB = it->tb();
      lastOL = 9999;
      tb = doc->createElement(X("tb"));
      tb->setAttribute(X("num"), X( IntToString(lastTB).c_str()));
    }
    if (lastOL != it->ol()) {
      if(ol) { tb->appendChild(ol); ol = 0; }
      lastOL = it->ol();
      ol = doc->createElement(X("ol"));
      ol->setAttribute(X("num"), X( IntToString(lastOL).c_str()));
    }
    DOMElement * lmd = doc->createElement(X("lmd"));
    lmd->setAttribute(X("dat"), X( IntToString(it->lmd().dat,1).c_str()));
    lmd->setAttribute(X("del"), X( IntToString(it->lmd().del).c_str()));
    lmd->setAttribute(X("eod"), X( IntToString(it->lmd().eod).c_str()));
    lmd->setAttribute(X("hp"),  X( IntToString(it->lmd().hp).c_str()));
    lmd->setAttribute(X("lb"),  X( IntToString(it->lmd().lb).c_str()));
    lmd->setAttribute(X("par"), X( IntToString(it->lmd().par).c_str()));
    lmd->setAttribute(X("raw"), X( IntToString(it->lmd().raw(),1).c_str()));
    ol->appendChild(lmd); lmd=0;
  } 
  if(ol) { tb->appendChild(ol); ol = 0; } 
  if(tb) { tc->appendChild(tb); tb = 0; } 
  if(tc) { bx->appendChild(tc); tc = 0; } 
  if(bx) { event->appendChild(bx); bx = 0; } 
  rootElem->appendChild(event);
}

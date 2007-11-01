#include "DQM/HcalMonitorClient/interface/HcalDQMDbInterface.h"

using namespace std;

namespace {
  template <class T> XMLCh* transcode (const T& fInput) {
    ostringstream ost;
    ost << fInput;
    return XMLString::transcode(ost.str().c_str());
  } 
}

DOMElement* HcalDQMDbInterface::createElement(DOMDocument* doc, DOMElement* parent, char* name){
  DOMElement*  elem = doc->createElement(XML(name));
  parent->appendChild(elem);
  return elem;
}

DOMElement* HcalDQMDbInterface::createElement(DOMDocument* doc, DOMElement* parent, char* name, char* value){
  DOMElement*  elem = createElement(doc,parent,name);
  elem->appendChild(doc->createTextNode(XML(value)));
  return elem;
}

DOMElement* HcalDQMDbInterface::createElement(DOMDocument* doc, DOMElement* parent, char* name, const char* value){
  DOMElement*  elem = createElement(doc,parent,name);
  elem->appendChild(doc->createTextNode(XML(value)));
  return elem;
}

DOMDocument* HcalDQMDbInterface::createDocument(){
  DOMImplementation* impl =  DOMImplementationRegistry::getDOMImplementation(XML("Core"));
  return impl->createDocument(0,XML("ROOT"),0);
}

void HcalDQMDbInterface::writeDocument(DOMDocument* doc, const char* xmlFile){
  DOMImplementation* impl =  DOMImplementationRegistry::getDOMImplementation(XML("Core"));
  DOMWriter *theSerializer = ((DOMImplementationLS*)impl)->createDOMWriter();
  if (theSerializer->canSetFeature(XMLUni::fgDOMWRTFormatPrettyPrint, true))
    theSerializer->setFeature(XMLUni::fgDOMWRTFormatPrettyPrint, true);
  if (theSerializer->canSetFeature(XMLUni::fgDOMWRTBOM, true))
    theSerializer->setFeature(XMLUni::fgDOMWRTBOM, true);
  XMLFormatTarget *myFormTarget = new LocalFileFormatTarget(xmlFile);
  theSerializer->writeNode(myFormTarget, *doc);
  delete theSerializer;
  delete myFormTarget;
}

DOMElement* HcalDQMDbInterface::createFooter(DOMDocument* doc,
					     unsigned long long fIovBegin, unsigned long long fIovEnd,
					     const char* fTagName, const char* fDetectorName, const char* fComment){
  
  DOMElement* parent = doc->getDocumentElement();
  DOMElement* elems = createElement(doc,parent,"ELEMENTS");
  DOMElement* dataset = createElement(doc, elems, "DATA_SET");
  dataset->setAttribute(transcode("id"), transcode("-1"));
  createIOV(doc,elems,fIovBegin,fIovEnd);
  createTag(doc,elems,fTagName,fDetectorName,fComment);

  DOMElement* maps = createElement(doc,parent,"MAPS");
  DOMElement* mapTag = makeMapTag(doc,maps);
  DOMElement* mapIov = makeMapIOV(doc,mapTag);
  makeMapDataset(doc,mapIov);

  return elems;
}

DOMElement* HcalDQMDbInterface::makeMapTag(DOMDocument* doc,DOMElement* fMap) {
  DOMElement* tag = createElement(doc, fMap, "TAG");
  tag->setAttribute(transcode("idref"), transcode("TAG_ID"));
  return tag;
}

DOMElement* HcalDQMDbInterface::makeMapIOV(DOMDocument* doc,DOMElement* fTag) {
  DOMElement* iov = createElement(doc,fTag, "IOV");
  iov->setAttribute(transcode("idref"), transcode("IOV_ID"));
  return iov;
}

DOMElement* HcalDQMDbInterface::makeMapDataset(DOMDocument* doc,DOMElement* fIov) {
  DOMElement* element = createElement(doc,fIov, "DATA_SET");
  element->setAttribute(transcode("idref"), transcode("-1"));
  return element;
}

DOMElement* HcalDQMDbInterface::createIOV(DOMDocument* doc,DOMElement*  parent, 
					  unsigned long long fIovBegin, unsigned long long fIovEnd) {
  DOMElement* iov = createElement(doc,parent,"IOV");
  iov->setAttribute(transcode("id"), transcode("IOV_ID"));
  
  createElement(doc,iov,"INTERVAL_OF_VALIDITY_BEGIN", itoa(fIovBegin));
  if(fIovEnd) {
    createElement(doc,iov,"INTERVAL_OF_VALIDITY_END", itoa(fIovEnd));
  }
  return iov;
}

DOMElement* HcalDQMDbInterface::createTag(DOMDocument* doc,DOMElement*  parent, 
					   const char* fTagName, const char* fDetectorName, const char* fComment) {
  DOMElement* tag = createElement(doc,parent,"TAG");
  tag->setAttribute(transcode("id"), transcode ("TAG_ID"));
  tag->setAttribute(transcode("mode"), transcode ("auto"));

  createElement(doc,tag,"TAG_NAME", fTagName);
  createElement(doc,tag,"DETECTOR_NAME", fDetectorName);
  createElement(doc,tag,"COMMENT_DESCRIPTION", fComment);

  return tag;
}


void HcalHotCellDbInterface::createHeader(DOMDocument* doc, unsigned int runno, const char* startTime){
  DOMElement*  parent = doc->getDocumentElement();
  DOMElement*  headerElem = createElement(doc,parent,"HEADER");
  DOMElement*  typeElem = createElement(doc,headerElem,"TYPE");
  createElement(doc,typeElem,"EXTENSION_TABLE_NAME","HCAL_CHANNEL_ON_OFF_STATES");
  createElement(doc,typeElem,"NAME","HCAL channel on off states");
  DOMElement*  runElem = createElement(doc,headerElem,"RUN");
  createElement(doc,runElem,"RUN_TYPE","hcal-dqm-onoff-test");
  createElement(doc,runElem,"RUN_NUMBER",itoa(runno));
  createElement(doc,runElem,"RUN_BEGIN_TIMESTAMP",startTime);
  createElement(doc,runElem,"COMMENT_DESCRIPTION","dqm data");
}

DOMElement* HcalDQMDbInterface::createChannel(DOMDocument* doc,DOMElement* parent, HcalDetId id){
  HcalText2DetIdConverter converter(id);
  DOMElement*  chanElem = createElement(doc,parent,"CHANNEL");
  createElement(doc,chanElem,"EXTENSION_TABLE_NAME","HCAL_CHANNELS");
  createElement(doc,chanElem,"ETA",itoa(id.ietaAbs()));
  createElement(doc,chanElem,"PHI",itoa(id.iphi()));
  createElement(doc,chanElem,"DEPTH",itoa(id.depth()));
  createElement(doc,chanElem,"Z",itoa(id.zside()));
  createElement(doc,chanElem,"DETECTOR_NAME",converter.getFlavor().c_str());
  createElement(doc,chanElem,"HCAL_CHANNEL_ID",itoa(id.rawId()));
  return chanElem;
}

DOMElement* HcalHotCellDbInterface::createData(DOMDocument* doc,DOMElement* parent, HcalDQMChannelQuality::Item item){
  DOMElement*  dataElem = createElement(doc,parent,"DATA");
  createElement(doc,dataElem,"CHANNEL_ON_OFF_STATE",itoa(item.mMasked));
  createElement(doc,dataElem,"CHANNEL_STATUS_WORD",itoa(item. mQuality));
  createElement(doc,dataElem,"COMMENT_DESCRIPTION",item.mComment.c_str());
  return dataElem;
}


void HcalHotCellDbInterface::createDataset(DOMDocument* doc,
					   HcalDQMChannelQuality::Item item,
					   const char* gmtime,
					   const char* version){

  DOMElement*  parent = doc->getDocumentElement();
  DOMElement*  dataSetElem = createElement(doc,parent,"DATA_SET");
  createElement(doc,dataSetElem,"VERSION",version);
  createElement(doc,dataSetElem,"CREATION_TIMESTAMP",gmtime);
  createElement(doc,dataSetElem,"CREATED_BY","wfisher");

  HcalDetId id(item.mId);
  createChannel(doc, dataSetElem, id);
  createData(doc, dataSetElem,item);
}


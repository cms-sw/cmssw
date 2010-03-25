#include "DQM/HcalMonitorClient/interface/HcalDQMDbInterface.h"

namespace {
  template <class T> XMLCh* transcode (const T& fInput) {
    std::ostringstream ost;
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
  DOMElement* elems = createElement(doc,parent,(char*)"ELEMENTS");
  DOMElement* dataset = createElement(doc, elems, (char*)"DATA_SET");
  dataset->setAttribute(transcode("id"), transcode("-1"));
  createIOV(doc,elems,fIovBegin,fIovEnd);
  createTag(doc,elems,fTagName,fDetectorName,fComment);

  DOMElement* maps = createElement(doc,parent,(char*)"MAPS");
  DOMElement* mapTag = makeMapTag(doc,maps);
  DOMElement* mapIov = makeMapIOV(doc,mapTag);
  makeMapDataset(doc,mapIov);

  return elems;
}

DOMElement* HcalDQMDbInterface::makeMapTag(DOMDocument* doc,DOMElement* fMap) {
  DOMElement* tag = createElement(doc, fMap, (char*)"TAG");
  tag->setAttribute(transcode("idref"), transcode("TAG_ID"));
  return tag;
}

DOMElement* HcalDQMDbInterface::makeMapIOV(DOMDocument* doc,DOMElement* fTag) {
  DOMElement* iov = createElement(doc,fTag, (char*)"IOV");
  iov->setAttribute(transcode("idref"), transcode("IOV_ID"));
  return iov;
}

DOMElement* HcalDQMDbInterface::makeMapDataset(DOMDocument* doc,DOMElement* fIov) {
  DOMElement* element = createElement(doc,fIov, (char*)"DATA_SET");
  element->setAttribute(transcode("idref"), transcode("-1"));
  return element;
}

DOMElement* HcalDQMDbInterface::createIOV(DOMDocument* doc,DOMElement*  parent, 
					  unsigned long long fIovBegin, unsigned long long fIovEnd) {
  DOMElement* iov = createElement(doc,parent,(char*)"IOV");
  iov->setAttribute(transcode("id"), transcode("IOV_ID"));
  
  createElement(doc,iov,(char*)"INTERVAL_OF_VALIDITY_BEGIN", itoa(fIovBegin));
  if(fIovEnd) {
    createElement(doc,iov,(char*)"INTERVAL_OF_VALIDITY_END", itoa(fIovEnd));
  }
  return iov;
}

DOMElement* HcalDQMDbInterface::createTag(DOMDocument* doc,DOMElement*  parent, 
					   const char* fTagName, const char* fDetectorName, const char* fComment) {
  DOMElement* tag = createElement(doc,parent,(char*)"TAG");
  tag->setAttribute(transcode("id"), transcode ("TAG_ID"));
  tag->setAttribute(transcode("mode"), transcode ("auto"));

  createElement(doc,tag,(char*)"TAG_NAME", fTagName);
  createElement(doc,tag,(char*)"DETECTOR_NAME", fDetectorName);
  createElement(doc,tag,(char*)"COMMENT_DESCRIPTION", fComment);

  return tag;
}


void HcalHotCellDbInterface::createHeader(DOMDocument* doc, unsigned int runno, const char* startTime){
  DOMElement*  parent = doc->getDocumentElement();
  DOMElement*  headerElem = createElement(doc,parent,(char*)"HEADER");
  DOMElement*  typeElem = createElement(doc,headerElem,(char*)"TYPE");
  createElement(doc,typeElem,(char*)"EXTENSION_TABLE_NAME",(char*)"HCAL_CHANNEL_ON_OFF_STATES");
  createElement(doc,typeElem,(char*)"NAME",(char*)"HCAL channel on off states");
  DOMElement*  runElem = createElement(doc,headerElem,(char*)"RUN");
  createElement(doc,runElem,(char*)"RUN_TYPE",(char*)"hcal-dqm-onoff-test");
  createElement(doc,runElem,(char*)"RUN_NUMBER",itoa(runno));
  createElement(doc,runElem,(char*)"RUN_BEGIN_TIMESTAMP",startTime);
  createElement(doc,runElem,(char*)"COMMENT_DESCRIPTION",(char*)"dqm data");
}

DOMElement* HcalDQMDbInterface::createChannel(DOMDocument* doc,DOMElement* parent, HcalDetId id){
  HcalText2DetIdConverter converter(id);
  DOMElement*  chanElem = createElement(doc,parent,(char*)"CHANNEL");
  createElement(doc,chanElem,(char*)"EXTENSION_TABLE_NAME",(char*)"HCAL_CHANNELS");
  createElement(doc,chanElem,(char*)"ETA",itoa(id.ietaAbs()));
  createElement(doc,chanElem,(char*)"PHI",itoa(id.iphi()));
  createElement(doc,chanElem,(char*)"DEPTH",itoa(id.depth()));
  createElement(doc,chanElem,(char*)"Z",itoa(id.zside()));
  createElement(doc,chanElem,(char*)"DETECTOR_NAME",converter.getFlavor().c_str());
  createElement(doc,chanElem,(char*)"HCAL_CHANNEL_ID",itoa(id.rawId()));
  return chanElem;
}

DOMElement* HcalHotCellDbInterface::createData(DOMDocument* doc,DOMElement* parent, HcalDQMChannelQuality::Item item){
  DOMElement*  dataElem = createElement(doc,parent,(char*)"DATA");
  createElement(doc,dataElem,(char*)"CHANNEL_ON_OFF_STATE",itoa(item.mMasked));
  createElement(doc,dataElem,(char*)"CHANNEL_STATUS_WORD",itoa(item. mQuality));
  createElement(doc,dataElem,(char*)"COMMENT_DESCRIPTION",item.mComment.c_str());
  return dataElem;
}


void HcalHotCellDbInterface::createDataset(DOMDocument* doc,
					   HcalDQMChannelQuality::Item item,
					   const char* gmtime,
					   const char* version){

  DOMElement*  parent = doc->getDocumentElement();
  DOMElement*  dataSetElem = createElement(doc,parent,(char*)"DATA_SET");
  createElement(doc,dataSetElem,(char*)"VERSION",version);
  createElement(doc,dataSetElem,(char*)"CREATION_TIMESTAMP",gmtime);
  createElement(doc,dataSetElem,(char*)"CREATED_BY",(char*)"wfisher");

  HcalDetId id(item.mId);
  createChannel(doc, dataSetElem, id);
  createData(doc, dataSetElem,item);
}

void HcalHLXMaskDbInterface::createHeader(DOMDocument* doc){
  DOMElement*  parent = doc->getDocumentElement();
  DOMElement*  headerElem = createElement(doc,parent,(char*)"HEADER");
  DOMElement*  typeElem = createElement(doc,headerElem,(char*)"TYPE");
  createElement(doc,typeElem,(char*)"EXTENSION_TABLE_NAME",(char*)"HCAL_HLX_MASKS_TYPE01");
  createElement(doc,typeElem,(char*)"NAME",(char*)"HCAL HLX masks [type 1]");
  DOMElement* element= createElement(doc,headerElem,(char*)"RUN");
  element->setAttribute(transcode("mode"), transcode("no-run"));
}

void HcalHLXMaskDbInterface::createData(DOMDocument* doc,DOMElement* parent, HcalHLXMask masks){
  DOMElement*  dataElem = createElement(doc,parent,(char*)"DATA");
  createElement(doc, dataElem, (char*)"FPGA", masks.position);
  char tmp[5] = "fooo";
  sprintf(tmp,"%i",masks.occMask);
  createElement(doc, dataElem, (char*)"OCC_MASK", tmp);
  sprintf(tmp,"%i",masks.lhcMask);
  createElement(doc, dataElem, (char*)"LHC_MASK", tmp);
  sprintf(tmp,"%i",masks.sumEtMask);
  createElement(doc, dataElem, (char*)"SUM_ET_MASK", tmp);
}

DOMElement* HcalHLXMaskDbInterface::createDataset(DOMDocument* doc,
						  const HcalHLXMask masks,
						  const char* gmtime,
						  const char* version, const char* subversion){

  DOMElement*  parent = doc->getDocumentElement();
  DOMElement*  dataSetElem = createElement(doc,parent,(char*)"DATA_SET");
  createElement(doc,dataSetElem,(char*)"VERSION",version);
  createElement(doc,dataSetElem,(char*)"SUBVERSION",subversion);
  createElement(doc,dataSetElem,(char*)"CREATION_TIMESTAMP",gmtime);
  createElement(doc,dataSetElem,(char*)"CREATED_BY",(char*)"jwerner");

  DOMElement*  partAssElem = createElement(doc,dataSetElem,(char*)"PART_ASSEMBLY");
  DOMElement* parentPartAssElem = createElement(doc,partAssElem,(char*)"PARENT_PART");
  createElement(doc, parentPartAssElem, (char*)"KIND_OF_PART", (char*)"HCAL HTR Crate");
  char tmp[5];
  if(masks.crateId <10){ sprintf(tmp,"CRATE0%i",masks.crateId);}
  else{ sprintf(tmp,"CRATE%i",masks.crateId);}
  createElement(doc, parentPartAssElem, (char*)"NAME_LABEL",tmp);
  //end PARENT_PART 
  DOMElement* childUniqueIdByElem = createElement(doc,partAssElem,(char*)"CHILD_UNIQUELY_IDENTIFIED_BY");
  createElement(doc, childUniqueIdByElem, (char*)"KIND_OF_PART", (char*)"HCAL HTR Crate Slot");
  DOMElement* attributeElem = createElement(doc,childUniqueIdByElem,(char*)"ATTRIBUTE");
  createElement(doc, attributeElem, (char*)"NAME", (char*)"HCAL HTR Slot Number");
  createElement(doc, attributeElem, (char*)"VALUE", itoa(masks.slotId));
  //end attribute                                                                                                                  
  //end child uni...                                                                                                               
  //end part assembly                                                                                                              

  return dataSetElem;
}

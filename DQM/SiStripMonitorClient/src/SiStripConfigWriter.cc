#include "DQM/SiStripMonitorClient/interface/SiStripConfigWriter.h"


#include "DQMServices/ClientConfig/interface/ParserFunctions.h"

//
// -- Constructor
// 
SiStripConfigWriter::SiStripConfigWriter() {
}
//
// -- Destructor
//
SiStripConfigWriter::~SiStripConfigWriter() {

}
//
// -- Initialize XML
// 
bool SiStripConfigWriter::init(std::string main) {
  cms::concurrency::xercesInitialize();

  xercesc::DOMImplementation* domImpl = xercesc::DOMImplementationRegistry::getDOMImplementation(qtxml::_toDOMS("Range"));
  if (!domImpl) return false;
  domWriter = (dynamic_cast<xercesc::DOMImplementation*>(domImpl))->createDOMWriter();
  if (!domWriter) return false;
  domWriter->canSetFeature(xercesc::XMLUni::fgDOMWRTFormatPrettyPrint, true);
  theDoc = domImpl->createDocument(0,qtxml::_toDOMS(main), 0);
  if (!theDoc) return false;
  theTopElement = theDoc->getDocumentElement();
  return true;
}
//
// -- Add an Element to the top node
// 
void SiStripConfigWriter::createElement(std::string tag) {
  theTopElement->appendChild(theDoc->createTextNode(qtxml::_toDOMS("\n")));
  lastElement = theDoc->createElement(qtxml::_toDOMS(tag));
  theTopElement->appendChild(lastElement);
}
//
// -- Add an Element to the top node
// 
void SiStripConfigWriter::createElement(std::string tag, std::string name) {
  theTopElement->appendChild(theDoc->createTextNode(qtxml::_toDOMS("\n")));
  lastElement = theDoc->createElement(qtxml::_toDOMS(tag));
  lastElement->setAttribute(qtxml::_toDOMS("name"), qtxml::_toDOMS(name));
  theTopElement->appendChild(lastElement);
}
//
// -- Add a child to the last element
// 
void SiStripConfigWriter::createChildElement(std::string tag, std::string name) {

  lastElement->appendChild(theDoc->createTextNode(qtxml::_toDOMS("\n")));
  xercesc::DOMElement* newElement = theDoc->createElement(qtxml::_toDOMS(tag));
  newElement->setAttribute(qtxml::_toDOMS("name"), qtxml::_toDOMS(name));
  lastElement->appendChild(newElement);
}
//
// -- Add a child to the last element
// 
void SiStripConfigWriter::createChildElement(std::string tag, std::string name, std::string att_name, std::string att_val) {

  lastElement->appendChild(theDoc->createTextNode(qtxml::_toDOMS("\n")));
  xercesc::DOMElement* newElement = theDoc->createElement(qtxml::_toDOMS(tag));
  newElement->setAttribute(qtxml::_toDOMS("name"), qtxml::_toDOMS(name));
  newElement->setAttribute(qtxml::_toDOMS(att_name), qtxml::_toDOMS(att_val));

  lastElement->appendChild(newElement);

}
//
// -- Add a child to the last element
// 
void SiStripConfigWriter::createChildElement(std::string tag,std::string name,std::string att_name1, std::string att_val1,
                                                                      std::string att_name2, std::string att_val2) {
  lastElement->appendChild(theDoc->createTextNode(qtxml::_toDOMS("\n")));
  xercesc::DOMElement* newElement = theDoc->createElement(qtxml::_toDOMS(tag));
  newElement->setAttribute(qtxml::_toDOMS("name"), qtxml::_toDOMS(name));
  newElement->setAttribute(qtxml::_toDOMS(att_name1), qtxml::_toDOMS(att_val1));  
  newElement->setAttribute(qtxml::_toDOMS(att_name2), qtxml::_toDOMS(att_val2));
  lastElement->appendChild(newElement);

}
//
// -- Add a child to the last element
// 
void SiStripConfigWriter::createChildElement(std::string tag,std::string name,std::string att_name1, std::string att_val1,
                           					      std::string att_name2, std::string att_val2,
                                                                      std::string att_name3, std::string att_val3) {

  lastElement->appendChild(theDoc->createTextNode(qtxml::_toDOMS("\n")));
  xercesc::DOMElement* newElement = theDoc->createElement(qtxml::_toDOMS(tag));
  newElement->setAttribute(qtxml::_toDOMS("name"), qtxml::_toDOMS(name));
  newElement->setAttribute(qtxml::_toDOMS(att_name1), qtxml::_toDOMS(att_val1));  
  newElement->setAttribute(qtxml::_toDOMS(att_name2), qtxml::_toDOMS(att_val2));
  newElement->setAttribute(qtxml::_toDOMS(att_name3), qtxml::_toDOMS(att_val3));
  lastElement->appendChild(newElement);

}
//
// -- Write to File
// 
void SiStripConfigWriter::write(std::string fname) {
  xercesc::XMLFormatTarget* formTarget = new xercesc::LocalFileFormatTarget(fname.c_str());
  domWriter->writeNode(formTarget, *theTopElement);
  delete formTarget;
  theDoc->release(); 
}

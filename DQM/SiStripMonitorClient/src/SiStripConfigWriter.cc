#include "DQM/SiStripMonitorClient/interface/SiStripConfigWriter.h"


#include "DQMServices/ClientConfig/interface/ParserFunctions.h"

using namespace xercesc;
using namespace std;
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
bool SiStripConfigWriter::init(string main) {
  try {
    XMLPlatformUtils::Initialize();
  }
  catch (const XMLException& toCatch) {
    cout << "Problem to initialise XML !!! " << endl;
    return false;
  }
  DOMImplementation* domImpl = DOMImplementationRegistry::getDOMImplementation(qtxml::_toDOMS("Range"));
  domWriter = (dynamic_cast<DOMImplementation*>(domImpl))->createDOMWriter();
  domWriter->canSetFeature(XMLUni::fgDOMWRTFormatPrettyPrint, true);
  theDoc = domImpl->createDocument(0,qtxml::_toDOMS(main), 0);
  theTopElement = theDoc->getDocumentElement();
  return true;
}
//
// -- Add an Element to the top node
// 
void SiStripConfigWriter::createElement(string tag) {
  theTopElement->appendChild(theDoc->createTextNode(qtxml::_toDOMS("\n")));
  lastElement = theDoc->createElement(qtxml::_toDOMS(tag));
  theTopElement->appendChild(lastElement);
}
//
// -- Add an Element to the top node
// 
void SiStripConfigWriter::createElement(string tag, string name) {
  theTopElement->appendChild(theDoc->createTextNode(qtxml::_toDOMS("\n")));
  lastElement = theDoc->createElement(qtxml::_toDOMS(tag));
  lastElement->setAttribute(qtxml::_toDOMS("name"), qtxml::_toDOMS(name));
  theTopElement->appendChild(lastElement);
}
//
// -- Add a child to the last element
// 
void SiStripConfigWriter::createChildElement(string tag, string name) {

  lastElement->appendChild(theDoc->createTextNode(qtxml::_toDOMS("\n")));
  xercesc::DOMElement* newElement = theDoc->createElement(qtxml::_toDOMS(tag));
  newElement->setAttribute(qtxml::_toDOMS("name"), qtxml::_toDOMS(name));
  lastElement->appendChild(newElement);
}
//
// -- Add a child to the last element
// 
void SiStripConfigWriter::createChildElement(string tag, string name, string att_name, string att_val) {

  lastElement->appendChild(theDoc->createTextNode(qtxml::_toDOMS("\n")));
  xercesc::DOMElement* newElement = theDoc->createElement(qtxml::_toDOMS(tag));
  newElement->setAttribute(qtxml::_toDOMS("name"), qtxml::_toDOMS(name));
  newElement->setAttribute(qtxml::_toDOMS(att_name), qtxml::_toDOMS(att_val));

  lastElement->appendChild(newElement);

}
//
// -- Add a child to the last element
// 
void SiStripConfigWriter::createChildElement(string tag,string name,string att_name1, string att_val1,
                                                                      string att_name2, string att_val2) {
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
void SiStripConfigWriter::createChildElement(string tag,string name,string att_name1, string att_val1,
                           					      string att_name2, string att_val2,
                                                                      string att_name3, string att_val3) {

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
void SiStripConfigWriter::write(string fname) {
  XMLFormatTarget* formTarget = new LocalFileFormatTarget(fname.c_str());
  domWriter->writeNode(formTarget, *theTopElement);
  delete formTarget;
  theDoc->release(); 
}

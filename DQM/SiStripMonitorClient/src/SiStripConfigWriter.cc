#include "DQM/SiStripMonitorClient/interface/SiStripConfigWriter.h"
#include "DQMServices/ClientConfig/interface/ParserFunctions.h"
#include <memory>

using namespace std;
using namespace xercesc;

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

  unique_ptr<DOMImplementation> domImpl( DOMImplementationRegistry::getDOMImplementation(qtxml::_toDOMS("Range")));
  if( domImpl == nullptr ) return false;
  theDomWriter = domImpl->createLSSerializer();
  if( theDomWriter == nullptr ) return false;
  if( theDomWriter->getDomConfig()->canSetParameter( XMLUni::fgDOMWRTFormatPrettyPrint, true ))
    theDomWriter->getDomConfig()->setParameter( XMLUni::fgDOMWRTFormatPrettyPrint, true );
  theDoc = domImpl->createDocument( 0, qtxml::_toDOMS(main), 0 );
  if( theDoc == nullptr ) return false;
  theTopElement = theDoc->getDocumentElement();
  theOutput = domImpl->createLSOutput();
  if( theOutput == nullptr ) return false;
  return true;
}
//
// -- Add an Element to the top node
// 
void SiStripConfigWriter::createElement(std::string tag) {
  theTopElement->appendChild(theDoc->createTextNode(qtxml::_toDOMS("\n")));
  theLastElement = theDoc->createElement(qtxml::_toDOMS(tag));
  theTopElement->appendChild(theLastElement);
}
//
// -- Add an Element to the top node
// 
void SiStripConfigWriter::createElement(std::string tag, std::string name) {
  theTopElement->appendChild(theDoc->createTextNode(qtxml::_toDOMS("\n")));
  theLastElement = theDoc->createElement(qtxml::_toDOMS(tag));
  theLastElement->setAttribute(qtxml::_toDOMS("name"), qtxml::_toDOMS(name));
  theTopElement->appendChild(theLastElement);
}
//
// -- Add a child to the last element
// 
void SiStripConfigWriter::createChildElement(std::string tag, std::string name) {

  theLastElement->appendChild(theDoc->createTextNode(qtxml::_toDOMS("\n")));
  DOMElement* newElement = theDoc->createElement(qtxml::_toDOMS(tag));
  newElement->setAttribute(qtxml::_toDOMS("name"), qtxml::_toDOMS(name));
  theLastElement->appendChild(newElement);
}
//
// -- Add a child to the last element
// 
void SiStripConfigWriter::createChildElement(std::string tag, std::string name, std::string att_name, std::string att_val) {

  theLastElement->appendChild(theDoc->createTextNode(qtxml::_toDOMS("\n")));
  DOMElement* newElement = theDoc->createElement(qtxml::_toDOMS(tag));
  newElement->setAttribute(qtxml::_toDOMS("name"), qtxml::_toDOMS(name));
  newElement->setAttribute(qtxml::_toDOMS(att_name), qtxml::_toDOMS(att_val));

  theLastElement->appendChild(newElement);

}
//
// -- Add a child to the last element
// 
void SiStripConfigWriter::createChildElement(std::string tag,std::string name,std::string att_name1, std::string att_val1,
                                                                      std::string att_name2, std::string att_val2) {
  theLastElement->appendChild(theDoc->createTextNode(qtxml::_toDOMS("\n")));
  DOMElement* newElement = theDoc->createElement(qtxml::_toDOMS(tag));
  newElement->setAttribute(qtxml::_toDOMS("name"), qtxml::_toDOMS(name));
  newElement->setAttribute(qtxml::_toDOMS(att_name1), qtxml::_toDOMS(att_val1));  
  newElement->setAttribute(qtxml::_toDOMS(att_name2), qtxml::_toDOMS(att_val2));
  theLastElement->appendChild(newElement);

}
//
// -- Add a child to the last element
// 
void SiStripConfigWriter::createChildElement(std::string tag,std::string name,std::string att_name1, std::string att_val1,
                           					      std::string att_name2, std::string att_val2,
                                                                      std::string att_name3, std::string att_val3) {

  theLastElement->appendChild(theDoc->createTextNode(qtxml::_toDOMS("\n")));
  DOMElement* newElement = theDoc->createElement(qtxml::_toDOMS(tag));
  newElement->setAttribute(qtxml::_toDOMS("name"), qtxml::_toDOMS(name));
  newElement->setAttribute(qtxml::_toDOMS(att_name1), qtxml::_toDOMS(att_val1));  
  newElement->setAttribute(qtxml::_toDOMS(att_name2), qtxml::_toDOMS(att_val2));
  newElement->setAttribute(qtxml::_toDOMS(att_name3), qtxml::_toDOMS(att_val3));
  theLastElement->appendChild(newElement);

}
//
// -- Write to File
// 
void SiStripConfigWriter::write(std::string fname) {
  XMLFormatTarget* formTarget = new LocalFileFormatTarget(fname.c_str());
  theOutput->setByteStream(formTarget);
  theDomWriter->write(theTopElement, theOutput);
  delete formTarget;
  theOutput->release();
  theDoc->release();
  theDomWriter->release();
}

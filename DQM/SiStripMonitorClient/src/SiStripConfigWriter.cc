#include "DQM/SiStripMonitorClient/interface/SiStripConfigWriter.h"


#include <iostream>
using namespace xercesc;
#include "DQMServices/ClientConfig/interface/ParserFunctions.h"

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
bool SiStripConfigWriter::init() {
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
  theDoc = domImpl->createDocument(0,qtxml::_toDOMS("Layouts"), 0);
  theTopElement = theDoc->getDocumentElement();
  theTopElement->appendChild(theDoc->createTextNode(qtxml::_toDOMS("\n")));
  return true;
}
//
// -- Add an Element
// 
void SiStripConfigWriter::createLayout(string& name) {
  lastLayout = theDoc->createElement(qtxml::_toDOMS("layout"));
  lastLayout->setAttribute(qtxml::_toDOMS("name"), qtxml::_toDOMS(name));
  theTopElement->appendChild(lastLayout);
}
//
// -- Add an Element
// 
void SiStripConfigWriter::createRow() {
  lastLayout->appendChild(theDoc->createTextNode(qtxml::_toDOMS("\n")));

  lastRow = theDoc->createElement(qtxml::_toDOMS("row"));
  lastLayout->appendChild(lastRow);
  lastLayout->appendChild(theDoc->createTextNode(qtxml::_toDOMS("\n")));
}
//
// -- Add an Element with Children
//
void SiStripConfigWriter::createColumn(string& element, string& name) {

   lastRow->appendChild(theDoc->createTextNode(qtxml::_toDOMS("\n")));
   DOMElement* e1 = theDoc->createElement(qtxml::_toDOMS("column"));
   lastRow->appendChild(e1);

 
   DOMElement* e2 = theDoc->createElement(qtxml::_toDOMS(element));
   e2->setAttribute(qtxml::_toDOMS("name"),qtxml::_toDOMS(name));
   e1->appendChild(e2);
 }
//
// -- Write to File
// 
void SiStripConfigWriter::write(string& fname) {
  XMLFormatTarget* formTarget = new LocalFileFormatTarget(fname.c_str());
  domWriter->writeNode(formTarget, *theTopElement);
  delete formTarget;
  theDoc->release(); 


}

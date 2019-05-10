#include "DQM/SiPixelMonitorClient/interface/SiPixelConfigWriter.h"

#include "DQMServices/ClientConfig/interface/ParserFunctions.h"
#include <memory>

using namespace xercesc;
using namespace std;
//
// -- Constructor
//
SiPixelConfigWriter::SiPixelConfigWriter() {}
//
// -- Destructor
//
SiPixelConfigWriter::~SiPixelConfigWriter() {}
//
// -- Initialize XML
//
bool SiPixelConfigWriter::init() {
  try {
    cms::concurrency::xercesInitialize();
  } catch (const XMLException &toCatch) {
    cout << "Problem to initialise XML !!! " << endl;
    return false;
  }
  unique_ptr<DOMImplementation> domImpl(DOMImplementationRegistry::getDOMImplementation(qtxml::_toDOMS("Range")));
  if (domImpl == nullptr)
    return false;
  theDomWriter = domImpl->createLSSerializer();
  if (theDomWriter == nullptr)
    return false;
  if (theDomWriter->getDomConfig()->canSetParameter(XMLUni::fgDOMWRTFormatPrettyPrint, true))
    theDomWriter->getDomConfig()->setParameter(XMLUni::fgDOMWRTFormatPrettyPrint, true);
  theDoc = domImpl->createDocument(nullptr, qtxml::_toDOMS("Layouts"), nullptr);
  if (theDoc == nullptr)
    return false;
  theTopElement = theDoc->getDocumentElement();
  theTopElement->appendChild(theDoc->createTextNode(qtxml::_toDOMS("\n")));
  theOutput = domImpl->createLSOutput();
  if (theOutput == nullptr)
    return false;
  return true;
}
//
// -- Add an Element
//
void SiPixelConfigWriter::createLayout(string &name) {
  theLastLayout = theDoc->createElement(qtxml::_toDOMS("layout"));
  theLastLayout->setAttribute(qtxml::_toDOMS("name"), qtxml::_toDOMS(name));
  theTopElement->appendChild(theLastLayout);
}
//
// -- Add an Element
//
void SiPixelConfigWriter::createRow() {
  theLastLayout->appendChild(theDoc->createTextNode(qtxml::_toDOMS("\n")));

  theLastRow = theDoc->createElement(qtxml::_toDOMS("row"));
  theLastLayout->appendChild(theLastRow);
  theLastLayout->appendChild(theDoc->createTextNode(qtxml::_toDOMS("\n")));
}
//
// -- Add an Element with Children
//
void SiPixelConfigWriter::createColumn(string &element, string &name) {
  theLastRow->appendChild(theDoc->createTextNode(qtxml::_toDOMS("\n")));
  DOMElement *e1 = theDoc->createElement(qtxml::_toDOMS("column"));
  theLastRow->appendChild(e1);

  DOMElement *e2 = theDoc->createElement(qtxml::_toDOMS(element));
  e2->setAttribute(qtxml::_toDOMS("name"), qtxml::_toDOMS(name));
  e1->appendChild(e2);
}
//
// -- Write to File
//
void SiPixelConfigWriter::write(string &fname) {
  XMLFormatTarget *formTarget = new LocalFileFormatTarget(fname.c_str());
  theOutput->setByteStream(formTarget);
  theDomWriter->write(theTopElement, theOutput);
  delete formTarget;
  theOutput->release();
  theDoc->release();
  theDomWriter->release();
}

#ifndef IORawData_RPCFileReader_XMLDataIO_H
#define IORawData_RPCFileReader_XMLDataIO_H

#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/framework/LocalFileFormatTarget.hpp>
XERCES_CPP_NAMESPACE_USE

namespace edm { class Event; }
class OptoTBData;

#include <string>
#include <sstream>
#include <vector>


class XStr {
public:
  XStr(const char* const toTranscode) { fUnicodeForm = XMLString::transcode(toTranscode); }
  ~XStr() { XMLString::release(&fUnicodeForm); }
  const XMLCh* unicodeForm() const { return fUnicodeForm; }
private:
  XMLCh*   fUnicodeForm;
};
#define X(str) XStr(str).unicodeForm()


class XMLDataIO {
public:
  XMLDataIO(const std::string &fileName); 
  virtual ~XMLDataIO();
  void write(const edm::Event& ev, const std::vector<OptoTBData> & optoData);
private:
  std::string IntToString(int i, int opt=0) {
    std::stringstream ss; if(opt==1) ss << std::hex << i << std::dec; else ss << i ;
    return ss.str();
  }

private:
  DOMWriter*  theSerializer;
  XMLFormatTarget *myFormTarget;
  DOMDocument* doc;
  DOMElement* rootElem;
};
#endif


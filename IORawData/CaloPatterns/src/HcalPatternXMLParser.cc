#include "xercesc/sax2/SAX2XMLReader.hpp"
#include <xercesc/sax2/XMLReaderFactory.hpp>
#include "xercesc/sax2/DefaultHandler.hpp"
#include "xercesc/sax2/Attributes.hpp"
#include "FWCore/Utilities/interface/Exception.h"
#include <xercesc/framework/MemBufInputSource.hpp>
XERCES_CPP_NAMESPACE_USE
#include "IORawData/CaloPatterns/interface/HcalPatternXMLParser.h"
#include <ostream>

class HcalPatternXMLParserImpl {
public:
  std::unique_ptr<SAX2XMLReader> parser;
};

HcalPatternXMLParser::HcalPatternXMLParser() {
  m_parser=nullptr;
}
HcalPatternXMLParser::~HcalPatternXMLParser() {
  if (m_parser!=nullptr) delete m_parser;
}

  /*
    Example
    <pre>
    <CFGBrick>
      <Parameter name='IETA' type='int'>14</Parameter>
      <Parameter name='IPHI' type='int'>2</Parameter>
      <Parameter name='DEPTH' type='int'>1</Parameter>
      <Parameter name='CRATE' type='int'>1</Parameter>
      <Parameter name='SLOT' type='int'>8</Parameter>
      <Parameter name='TOPBOTTOM' type='int'>0</Parameter>
      <Parameter name='CHANNEL' type='int'>6</Parameter>
      <Parameter name='LUT_TYPE' type='int'>1</Parameter>
      <Parameter name='CREATIONTAG' type='string'>Identity</Parameter>
      <Parameter name='CREATIONSTAMP' type='string'>2005-03-08 11:44:34</Parameter>
      <Parameter name='FORMATREVISION' type='string'>1</Parameter>
      <Parameter name='TARGETFIRMWARE' type='string'>0</Parameter>
      <Parameter name='GENERALIZEDINDEX' type='int'>140211</Parameter>
      <Data elements='128' encoding='hex'> .. </Data>
    </CFGBrick>
    </pre>
  */

  class ConfigurationDBHandler : public DefaultHandler {
    enum { md_Idle, md_Parameter, md_Data } m_mode;
  public:
    ConfigurationDBHandler(std::map<std::string,std::string>& parameters, std::vector<std::string>& items, std::string& encoding) : m_dataEncoding(encoding), m_items(items), m_parameters(parameters) {
      m_mode=md_Idle;
      xc_Parameter=XMLString::transcode("Parameter");
      xc_Data=XMLString::transcode("Data");
      xc_name=XMLString::transcode("name");
      xc_type=XMLString::transcode("type");
      xc_elements=XMLString::transcode("elements");      
      xc_encoding=XMLString::transcode("encoding");
      m_items.clear();
      m_parameters.clear();
    }
    ~ConfigurationDBHandler() override {
      XMLString::release(&xc_Parameter);
      XMLString::release(&xc_Data);
      XMLString::release(&xc_name);
      XMLString::release(&xc_type);
      XMLString::release(&xc_elements);
      XMLString::release(&xc_encoding);
    }
    void startElement (const XMLCh *const uri, const XMLCh *const localname, const XMLCh *const qname, const Attributes &attrs) override;
    void endElement (const XMLCh *const uri, const XMLCh *const localname, const XMLCh *const qname) override;
    void characters (const XMLCh *const chars, const XMLSize_t length) override;
    void ignorableWhitespace (const XMLCh *const chars, const XMLSize_t length) override;
  private:
    inline bool cvt2String(const XMLCh* val, std::string& ou) {
      if (val==nullptr) return false;
      char* tool=XMLString::transcode(val);
      ou=tool;
      XMLString::release(&tool);
      return true;
    }
    XMLCh *xc_Parameter, *xc_Data, *xc_name, *xc_type, *xc_elements, *xc_encoding;
    std::string m_pname, m_ptype, m_text;
    int n_elements;
    std::string& m_dataEncoding;
    std::vector<std::string>& m_items;
    std::map<std::string,std::string>& m_parameters;
    char m_workc[512];
    XMLCh m_workx[256];
  };

  void ConfigurationDBHandler::startElement (const XMLCh *const uri, const XMLCh *const localname, const XMLCh *const qname, const Attributes &attrs) {
    if (m_mode!=md_Idle) return; 
    if (!XMLString::compareIString(localname,xc_Parameter)) {
      // parameter name
      if (!cvt2String(attrs.getValue(xc_name),m_pname)) return;
      // parameter type
      if (!cvt2String(attrs.getValue(xc_type),m_ptype)) return;
      // switch mode
      m_mode=md_Parameter;
      m_text="";
    } else if (!XMLString::compareIString(localname,xc_Data)) {
      // elements
      std::string strElements;
      if (!cvt2String(attrs.getValue(xc_elements),strElements)) return;
      n_elements=atoi(strElements.c_str());
      // encoding
      m_dataEncoding="";
      cvt2String(attrs.getValue(xc_encoding),m_dataEncoding);
      // switch mode
      m_mode=md_Data;
      m_text="";
    }
   
  }
  void ConfigurationDBHandler::endElement (const XMLCh *const uri, const XMLCh *const localname, const XMLCh *const qname) {
    if (m_mode==md_Idle) return;

    if (m_mode==md_Parameter) {
      m_parameters[m_pname]=m_text; // ignore the type for now...
    } else if (m_mode==md_Data) {
      // parse the text
      std::string entry;
      for (std::string::iterator q=m_text.begin(); q!=m_text.end(); q++) {
	if (isspace(*q)) {
	  if (entry.empty()) continue;
	  m_items.push_back(entry);
	  entry="";
	} else entry+=*q;
      }
    }

    m_mode=md_Idle;
  }
  void ConfigurationDBHandler::ignorableWhitespace(const   XMLCh* chars, const XMLSize_t length) {
    if (m_mode==md_Idle) return;
    m_text+=' ';
  }
  void ConfigurationDBHandler::characters(const XMLCh* chars, const XMLSize_t length) {
    if (m_mode==md_Idle) return;
    unsigned int offset=0;
    while (offset<length) {
      unsigned int i=0;
      for (i=0; i<length-offset && i<255; i++) m_workx[i]=chars[i+offset];
      m_workx[i]=0; // terminate string
      XMLString::transcode(m_workx,m_workc,511);
      m_text+=m_workc;
      offset+=i;
    }
  }

void HcalPatternXMLParser::parse(const std::string& xmlDocument, std::map<std::string,std::string>& parameters, std::vector<std::string>& items, std::string& encoding) {
    // uses XERCES SAX2 parser
    ConfigurationDBHandler handler(parameters,items,encoding);
    
    try {
      if (m_parser==nullptr) {
	m_parser=new HcalPatternXMLParserImpl();
	m_parser->parser=std::unique_ptr<xercesc::SAX2XMLReader>(xercesc::XMLReaderFactory::createXMLReader());
      }
	 
      MemBufInputSource src((const unsigned char*)xmlDocument.c_str(), xmlDocument.length(),"hcal::PatternReader");
      m_parser->parser->setContentHandler(&handler);
      m_parser->parser->parse(src);
    } catch (std::exception& ex) {
      throw cms::Exception("ParseError") << ex.what();
    }
  }

void HcalPatternXMLParser::parse(const std::string& xmlDocument, std::map<std::string,std::string>& parameters, std::vector<uint32_t>& data) {
  std::vector<std::string> items;
  std::string encoding;

  this->parse(xmlDocument,parameters,items,encoding);
  int formatting=0;
  if (encoding=="dec") formatting=10;
  if (encoding=="hex") formatting=16;

  data.clear();
  for (std::vector<std::string>::const_iterator i=items.begin(); i!=items.end(); i++)
    data.push_back(strtol(i->c_str(),nullptr,formatting));

}

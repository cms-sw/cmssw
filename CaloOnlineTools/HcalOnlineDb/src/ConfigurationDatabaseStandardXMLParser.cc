#include "CaloOnlineTools/HcalOnlineDb/interface/ConfigurationDatabaseStandardXMLParser.hh"
#include <xercesc/sax2/SAX2XMLReader.hpp>
#include <xercesc/sax2/XMLReaderFactory.hpp>
#include "xercesc/sax2/DefaultHandler.hpp"
#include "xercesc/sax2/Attributes.hpp"

#include <iostream>
using namespace std;

ConfigurationDatabaseStandardXMLParser::ConfigurationDatabaseStandardXMLParser() : m_parser(nullptr) {
}
#include <xercesc/framework/MemBufInputSource.hpp>
XERCES_CPP_NAMESPACE_USE

  /*
    Example
    <pre>
    <CFGBrick>
      <Parameter name='IETA' type='int'>14</Parameter>
      <Parameter name='IPHI' type='int'>2</Parameter>
      <Parameter name='DEPTH' type='int'>1</Parameter>
      <Parameter name='CRATE' type='int'>1</Parameter>
      <Parameter name='SLOT' type='int'>8</Parameter>
      <Parameter name='TOPBOTTOM' type='int'>2</Parameter>
      <Parameter name='CHANNEL' type='int'>6</Parameter>
      <Parameter name='LUT_TYPE' type='int'>1</Parameter>
      <Parameter name='CREATIONTAG' type='string'>Identity</Parameter>
      <Parameter name='CREATIONSTAMP' type='string'>2005-03-08 11:44:34</Parameter>
      <Parameter name='FORMATREVISION' type='string'>1</Parameter>
      <Parameter name='TARGETFIRMWARE' type='string'>0</Parameter>
      <Parameter name='GENERALIZEDINDEX' type='int'>140211</Parameter>
      <Data elements='128' encoding='hex'> .. </Data>
      <Data elements='1' encoding='hex' rm=' ' card=' ' qie=' '></Data>
    </CFGBrick>
    </pre>
  */

  class ConfigurationDBHandler : public DefaultHandler {
    enum { md_Idle, md_Parameter, md_Data } m_mode;
  public:
    ConfigurationDBHandler(std::list<ConfigurationDatabaseStandardXMLParser::Item>& items) : m_items(items) {
      m_mode=md_Idle;
      xc_Parameter=XMLString::transcode("Parameter");
      xc_Data=XMLString::transcode("Data");
      xc_name=XMLString::transcode("name");
      xc_type=XMLString::transcode("type");
      xc_elements=XMLString::transcode("elements");      
      xc_encoding=XMLString::transcode("encoding");
      xc_header[0]=XMLString::transcode("CFGBrick");
      xc_header[1]=XMLString::transcode("LUT");
      xc_header[2]=XMLString::transcode("Pattern");
      m_items.clear();
    }
    ~ConfigurationDBHandler() override {
      XMLString::release(&xc_Parameter);
      XMLString::release(&xc_Data);
      XMLString::release(&xc_name);
      XMLString::release(&xc_type);
      XMLString::release(&xc_elements);
      XMLString::release(&xc_encoding);
      for (int i=0; i<ITEMELEMENTNAMES; i++) 
	XMLString::release(&xc_header[i]);
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
    static const int ITEMELEMENTNAMES=3;
    XMLCh *xc_header[ITEMELEMENTNAMES];
    std::string m_pname, m_ptype, m_text;
    int n_elements;
    ConfigurationDatabaseStandardXMLParser::Item m_workitem;
    std::list<ConfigurationDatabaseStandardXMLParser::Item>& m_items;
    //    std::string& m_dataEncoding;
    //  std::vector<std::string>& m_items;
    //std::map<std::string,std::string>& m_parameters;
    char m_workc[512];
    XMLCh m_workx[256];
    bool isItemElement(const XMLCh* const localname) {
      for (int i=0; i<ITEMELEMENTNAMES; i++)
	if (!XMLString::compareIString(localname,xc_header[i])) return true;
      return false;
    }
  };

  void ConfigurationDBHandler::startElement (const XMLCh *const uri, const XMLCh *const localname, const XMLCh *const qname, const Attributes &attrs) {
    if (m_mode!=md_Idle) return; 
    std::string work;
    cvt2String(localname,work);
    if (isItemElement(localname)) {
      m_workitem.parameters.clear();
      m_workitem.items.clear();
      m_workitem.encoding.clear();
    } else if (!XMLString::compareIString(localname,xc_Parameter)) {
      // parameter name
      if (!cvt2String(attrs.getValue(xc_name),m_pname)) return;
      // parameter type
      if (!cvt2String(attrs.getValue(xc_type),m_ptype)) return;
      // switch mode
      m_mode=md_Parameter;
      m_text="";
    } else if (!XMLString::compareIString(localname,xc_Data)) {
      m_workitem.items.clear();
      // elements
      std::string strElements;
      if (!cvt2String(attrs.getValue(xc_elements),strElements)) return;
      n_elements=atoi(strElements.c_str());
      // encoding
      m_workitem.encoding="";
      cvt2String(attrs.getValue(xc_encoding),m_workitem.encoding);
      // switch mode
      m_mode=md_Data;
      m_text="";
      // other attributes
      for (unsigned int jj=0; jj<attrs.getLength(); jj++) {
	if (!XMLString::compareIString(xc_elements,attrs.getValue(jj)) ||
	    !XMLString::compareIString(xc_encoding,attrs.getValue(jj))) 
	  continue; // already handled these two
	std::string atkey,atvalue;
	cvt2String(attrs.getLocalName(jj),atkey);
	cvt2String(attrs.getValue(jj),atvalue);
	m_workitem.parameters[atkey]=atvalue;
      }
    }
   
  }
  void ConfigurationDBHandler::endElement (const XMLCh *const uri, const XMLCh *const localname, const XMLCh *const qname) {
    if (m_mode==md_Idle) return;

    if (isItemElement(localname)) {
    } else if (m_mode==md_Parameter) {
      m_workitem.parameters[m_pname]=m_text; // ignore the type for now...
    } else if (m_mode==md_Data) {
      // parse the text
      std::string entry;
      for (std::string::iterator q=m_text.begin(); q!=m_text.end(); q++) {
	if (isspace(*q)) {
	  if (entry.empty()) continue;
	  m_workitem.items.push_back(entry);
	  entry="";
	} else entry+=*q;
      }
      if (!entry.empty()) m_workitem.items.push_back(entry);
      m_items.push_back(m_workitem); // save it
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

void ConfigurationDatabaseStandardXMLParser::parse(const std::string& xmlDocument, std::map<std::string,std::string>& parameters, std::vector<std::string>& items, std::string& encoding) noexcept(false) {
  // uses XERCES SAX2 parser
  std::list<Item> theItems;
  ConfigurationDBHandler handler(theItems);
    
  try {
    if (m_parser==nullptr) {
      m_parser=xercesc::XMLReaderFactory::createXMLReader();
    }
    
    MemBufInputSource src((const unsigned char*)xmlDocument.c_str(), xmlDocument.length(),"hcal::ConfigurationDatabase");
    m_parser->setContentHandler(&handler);
    m_parser->parse(src);
  } catch (std::exception& ex) {
    XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException,ex.what());
  }
  if (theItems.empty()) {
    XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException,"No data found");
  } else if (theItems.size()>1) {
    XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException,"Multiple items found");
  } else {
    parameters=theItems.front().parameters;
    items=theItems.front().items;
    encoding=theItems.front().encoding;
  }
}



void ConfigurationDatabaseStandardXMLParser::parseMultiple(const std::string& xmlDocument, std::list<Item>& items) noexcept(false) {
  // uses XERCES SAX2 parser
  ConfigurationDBHandler handler(items);
    
  try {
    if (m_parser==nullptr) {
      m_parser=xercesc::XMLReaderFactory::createXMLReader();
    }
    
    MemBufInputSource src((const unsigned char*)xmlDocument.c_str(), xmlDocument.length(),"hcal::ConfigurationDatabase");
    m_parser->setContentHandler(&handler);
    m_parser->parse(src);
  } catch (std::exception& ex) {
    XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException,ex.what());
  }
}

std::vector<unsigned int> ConfigurationDatabaseStandardXMLParser::Item::convert() const {
  std::vector<unsigned int> values;
  int strtol_base=0;
  if (encoding=="hex") strtol_base=16;
  else if (encoding=="dec") strtol_base=10;
      
  // convert the data
  for (unsigned int j=0; j<items.size(); j++) 
    values.push_back(strtol(items[j].c_str(),nullptr,strtol_base));
  return values;
}

/*******************************************************************************
*                                                                              *
*  Karol Bunkowski                                                             *
*  Warsaw University 2005                                                      *
*                                                                              *
*******************************************************************************/
#ifndef _STAND_ALONE
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#endif // _STAND_ALONE
#include "L1Trigger/RPCTrigger/interface/RPCPatternsParser.h"
#include <xercesc/sax2/SAX2XMLReader.hpp>
#include <xercesc/sax2/XMLReaderFactory.hpp>
#include <xercesc/util/PlatformUtils.hpp>
#include <sstream>
#include <fstream> 
#include <iostream>
#include "L1Trigger/RPCTrigger/interface/RPCException.h"
//#ifndef __BORLANDC__
//#include "xoap.h"
//#endif

using namespace xercesc;
using namespace std;

string xMLCh2String(const XMLCh* ch) {
#ifdef __BORLANDC__
  if(ch == 0) return "";
  WideString wstr(ch);
  AnsiString astr(wstr);
  return astr.c_str();
#else
	if(ch == 0) return "";

	//auto_ptr<char> v(XMLString::transcode(ch));
  //return string(v.get());
  char* buf = XMLString::transcode(ch);
  string str(buf);
  XMLString::release(&buf);
  return str;
#endif
}

const RPCPattern::RPCPatVec& RPCPatternsParser::getPatternsVec(const RPCConst::l1RpcConeCrdnts& coneCrds) const {
  TPatternsVecsMap::const_iterator patVecIt  = m_PatternsVecsMap.find(coneCrds);
  if(patVecIt == m_PatternsVecsMap.end()){

    std::stringstream ss;
    ss << coneCrds.m_Tower << " " << coneCrds.m_LogSector << " " << coneCrds.m_LogSegment;
    throw RPCException( std::string("no such a cone in m_PatternsVecsMap: ")+ ss.str() );
    //edm::LogError("RPCTrigger")<< "no such a cone in m_PatternsVecsMap";
  }
  return patVecIt->second; // XXX - TMF - was in if{}, changed to avoid warning
}

const RPCPattern::RPCPatVec& RPCPatternsParser::getPatternsVec(const int tower, const int sc, const int sg) const {

    RPCConst::l1RpcConeCrdnts cords(tower,sc,sg);

    return getPatternsVec(cords);

}


// ---------------------------------------------------------------------------
//  This is a simple class that lets us do easy(though not terribly efficient)
//  trancoding of char* data to XMLCh data.
// ---------------------------------------------------------------------------
class XStr
{
public :
    // -----------------------------------------------------------------------
    //  Constructors and Destructor
    // -----------------------------------------------------------------------
    XStr(const char* const toTranscode)
    {
        // Call the private transcoding method
        m_fUnicodeForm = XMLString::transcode(toTranscode);
    }

    ~XStr()
    {
        XMLString::release(&m_fUnicodeForm);
    }


    // -----------------------------------------------------------------------
    //  Getter methods
    // -----------------------------------------------------------------------
    const XMLCh* unicodeForm() const
    {
        return m_fUnicodeForm;
    }

private :
    // -----------------------------------------------------------------------
    //  Private data members
    //
    //  m_fUnicodeForm
    //      This is the Unicode XMLCh format of the string.
    // -----------------------------------------------------------------------
    XMLCh*   m_fUnicodeForm;
};

#define Char2XMLCh(str) XStr(str).unicodeForm()

int RPCPatternsParser::m_InstanceCount = 0;

RPCPatternsParser::RPCPatternsParser()
{
   if(m_InstanceCount == 0) { 
    try {
        XMLPlatformUtils::Initialize();
        //XPathEvaluator::initialize();
        m_InstanceCount++;
    }
    catch(const XMLException &toCatch)  {
      throw RPCException("Error during Xerces-c Initialization: "
           + xMLCh2String(toCatch.getMessage()));
      //edm::LogError("RPCTrigger")<< "Error during Xerces-c Initialization: " 
      //           + xMLCh2String(toCatch.getMessage());
    }
  }  
}



RPCPatternsParser::~RPCPatternsParser() {
}

void RPCPatternsParser::parse(std::string fileName)
{
  ifstream fin;
  fin.open(fileName.c_str());
  if(fin.fail()) {
    throw RPCException("Cannot open the file" + fileName);
    //edm::LogError("RPCTrigger") << "Cannot open the file" + fileName;
  }
  fin.close();

  SAX2XMLReader* parser = XMLReaderFactory::createXMLReader();
  parser->setContentHandler(this);
  
  m_QualityVec.clear();
  parser->parse(fileName.c_str());
  delete parser;
}

void RPCPatternsParser::startElement(const XMLCh* const uri,
                                       const XMLCh* const localname,
                                       const XMLCh* const qname,
                                       const Attributes& attrs) {
  RPCConst rpcconst;
  
  m_CurrElement = xMLCh2String(localname);
  if(m_CurrElement == "quality") {
    //<quality id = "0" planes = "011110" val = 1/>
    RPCPattern::TQuality quality;
    
    
    quality.m_QualityTabNumber = rpcconst.stringToInt(xMLCh2String(attrs.getValue(Char2XMLCh("id"))));
    std::bitset<8> firedPl( xMLCh2String(attrs.getValue(Char2XMLCh("planes")) )) ;
    unsigned long fpUL = firedPl.to_ulong();
    quality.m_FiredPlanes = (unsigned char) (fpUL & 0xFF );
    //quality.m_FiredPlanes = xMLCh2String(attrs.getValue(Char2XMLCh("planes")));
    quality.m_QualityValue = rpcconst.stringToInt(xMLCh2String(attrs.getValue(Char2XMLCh("val"))));

    m_QualityVec.push_back(quality);
  }
  else if(m_CurrElement == "pac") {
    //<pac id ="0" m_tower = "0" logSector = "0" logSegment = "0" descr = "">       
    RPCConst::l1RpcConeCrdnts cone;
    cone.m_Tower = rpcconst.stringToInt(xMLCh2String(attrs.getValue(Char2XMLCh("tower"))));
    cone.m_LogSector = rpcconst.stringToInt(xMLCh2String(attrs.getValue(Char2XMLCh("logSector"))));
    cone.m_LogSegment = rpcconst.stringToInt(xMLCh2String(attrs.getValue(Char2XMLCh("logSegment"))));
    pair <TPatternsVecsMap::iterator, bool> res = m_PatternsVecsMap.insert(TPatternsVecsMap::value_type(cone,
                                                                                                RPCPattern::RPCPatVec()));
    if(res.second == true)
      m_CurPacIt = res.first;
    else
      throw RPCException( std::string("m_PatternsVecsMap insertion failed - cone already exixsts?"));
      //edm::LogError("RPCTrigger") << "m_PatternsVecsMap insertion failed - cone already exixsts?";
  }
  else if(m_CurrElement == "pat") {
    //<pat type="E" grp="0" qual="0" sign="0" code="31" num="0">
    string pt = xMLCh2String(attrs.getValue(Char2XMLCh("type")));
    if(pt == "E")
      m_CurPattern.setPatternType(RPCPattern::PAT_TYPE_E);
    else if(pt == "T")
      m_CurPattern.setPatternType(RPCPattern::PAT_TYPE_T);
    else
      throw RPCException("unknown pattern type: " + pt);
      //edm::LogError("RPCTrigger") << "unknown pattern type: " + pt;

    m_CurPattern.setRefGroup(rpcconst.stringToInt(xMLCh2String(attrs.getValue(Char2XMLCh("grp")))));
    m_CurPattern.setQualityTabNumber(rpcconst.stringToInt(xMLCh2String(attrs.getValue(Char2XMLCh("qual")))));
    
    m_CurPattern.setSign(rpcconst.stringToInt(xMLCh2String(attrs.getValue(Char2XMLCh("sign")))));
    m_CurPattern.setCode(rpcconst.stringToInt(xMLCh2String(attrs.getValue(Char2XMLCh("code")))));
    m_CurPattern.setNumber(rpcconst.stringToInt(xMLCh2String(attrs.getValue(Char2XMLCh("num")))));
  }
  else if(m_CurrElement == "str") {
    //<logstrip plane="m_LOGPLANE1" from="32" to="32"/>
    int logPlane = rpcconst.stringToInt(xMLCh2String(attrs.getValue(Char2XMLCh("Pl"))));
    m_CurPattern.setStripFrom(logPlane, rpcconst.stringToInt(xMLCh2String(attrs.getValue(Char2XMLCh("f")))));
    m_CurPattern.setStripTo(logPlane, rpcconst.stringToInt(xMLCh2String(attrs.getValue(Char2XMLCh("t")))) + 1);
  }
}

void RPCPatternsParser::endElement(const XMLCh* const uri, const XMLCh* const localname, const XMLCh* const qname) {
  string element = xMLCh2String(localname);
  if(element == "pat") {
    m_CurPacIt->second.push_back(m_CurPattern);
  }
}

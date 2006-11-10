/*******************************************************************************
*                                                                              *
*  Karol Bunkowski                                                             *
*  Warsaw University 2005                                                      *
*                                                                              *
*******************************************************************************/
#ifndef _STAND_ALONE
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#endif // _STAND_ALONE
#include "L1Trigger/RPCTrigger/src/L1RpcPatternsParser.h"
#include <xercesc/sax2/SAX2XMLReader.hpp>
#include <xercesc/sax2/XMLReaderFactory.hpp>
#include <xercesc/util/PlatformUtils.hpp>
#include <sstream>
#include <fstream> 
#include <iostream>
#include "L1Trigger/RPCTrigger/src/RPCException.h"
//#ifndef __BORLANDC__
//#include "xoap.h"
//#endif

using namespace xercesc;
using namespace std;

string XMLCh2String (const XMLCh* ch) {
#ifdef __BORLANDC__
  if (ch == 0) return "";
  WideString wstr(ch);
  AnsiString astr(wstr);
  return astr.c_str();
#else
	if (ch == 0) return "";

	//auto_ptr<char> v(XMLString::transcode (ch));
  //return string(v.get());
  char* buf = XMLString::transcode(ch);
  string str(buf);
  XMLString::release(&buf);
  return str;
#endif
}

const L1RpcPatternsVec& L1RpcPatternsParser::GetPatternsVec(const L1RpcConst::L1RpcConeCrdnts& coneCrds) const {
  TPatternsVecsMap::const_iterator patVecIt  = PatternsVecsMap.find(coneCrds);
  if(patVecIt == PatternsVecsMap.end()){
    throw L1RpcException( std::string("no such a cone in PatternsVecsMap"));
    //edm::LogError("RPCTrigger")<< "no such a cone in PatternsVecsMap";
  }
  return patVecIt->second; // XXX - TMF - was in if{}, changed to avoid warning
}

// ---------------------------------------------------------------------------
//  This is a simple class that lets us do easy (though not terribly efficient)
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
        fUnicodeForm = XMLString::transcode(toTranscode);
    }

    ~XStr()
    {
        XMLString::release(&fUnicodeForm);
    }


    // -----------------------------------------------------------------------
    //  Getter methods
    // -----------------------------------------------------------------------
    const XMLCh* unicodeForm() const
    {
        return fUnicodeForm;
    }

private :
    // -----------------------------------------------------------------------
    //  Private data members
    //
    //  fUnicodeForm
    //      This is the Unicode XMLCh format of the string.
    // -----------------------------------------------------------------------
    XMLCh*   fUnicodeForm;
};

#define Char2XMLCh(str) XStr(str).unicodeForm()

int L1RpcPatternsParser::InstanceCount = 0;

L1RpcPatternsParser::L1RpcPatternsParser()
{
   if (InstanceCount == 0) { 
    try {
        XMLPlatformUtils::Initialize();
        //XPathEvaluator::initialize();
        InstanceCount++;
    }
    catch(const XMLException &toCatch)  {
      throw L1RpcException("Error during Xerces-c Initialization: "
           + XMLCh2String(toCatch.getMessage()));
      //edm::LogError("RPCTrigger")<< "Error during Xerces-c Initialization: " + XMLCh2String(toCatch.getMessage());
    }
  }  
}



L1RpcPatternsParser::~L1RpcPatternsParser() {
}

void L1RpcPatternsParser::Parse(std::string fileName)
{
  ifstream fin;
  fin.open(fileName.c_str());
  if (fin.fail()) {
    throw L1RpcException("Cannot open the file" + fileName);
    //edm::LogError("RPCTrigger") << "Cannot open the file" + fileName;
  }
  fin.close();

  SAX2XMLReader* parser = XMLReaderFactory::createXMLReader();
  parser->setContentHandler(this);
  
  QualityVec.clear();
  parser->parse(fileName.c_str());
  delete parser;
}

void L1RpcPatternsParser::startElement(const XMLCh* const uri,
                                       const XMLCh* const localname,
                                       const XMLCh* const qname,
                                       const Attributes& attrs) {
  L1RpcConst rpcconst;
  
  CurrElement = XMLCh2String(localname);
  if(CurrElement == "quality") {
    //<quality id = "0" planes = "011110" val = 1/>
    TQuality quality;
    
    
    quality.QualityTabNumber = rpcconst.StringToInt(XMLCh2String(attrs.getValue(Char2XMLCh("id"))));
    quality.FiredPlanes = XMLCh2String(attrs.getValue(Char2XMLCh("planes")));
    quality.QualityValue = rpcconst.StringToInt(XMLCh2String(attrs.getValue(Char2XMLCh("val"))));

    QualityVec.push_back(quality);
  }
  else if(CurrElement == "pac") {
    //<pac id ="0" tower = "0" logSector = "0" logSegment = "0" descr = "">       
    L1RpcConst::L1RpcConeCrdnts cone;
    cone.Tower = rpcconst.StringToInt(XMLCh2String(attrs.getValue(Char2XMLCh("tower"))));
    cone.LogSector = rpcconst.StringToInt(XMLCh2String(attrs.getValue(Char2XMLCh("logSector"))));
    cone.LogSegment = rpcconst.StringToInt(XMLCh2String(attrs.getValue(Char2XMLCh("logSegment"))));
    pair <TPatternsVecsMap::iterator, bool> res = PatternsVecsMap.insert(TPatternsVecsMap::value_type(cone, L1RpcPatternsVec()));
    if(res.second == true)
      CurPacIt = res.first;
    else
      throw L1RpcException( std::string("PatternsVecsMap insertion failed - cone already exixsts?"));
      //edm::LogError("RPCTrigger") << "PatternsVecsMap insertion failed - cone already exixsts?";
  }
  else if(CurrElement == "pat") {
    //<pat type="E" grp="0" qual="0" sign="0" code="31" num="0">
    string pt = XMLCh2String(attrs.getValue(Char2XMLCh("type")));
    if(pt == "E")
      CurPattern.SetPatternType(L1RpcConst::PAT_TYPE_E);
    else if(pt == "T")
      CurPattern.SetPatternType(L1RpcConst::PAT_TYPE_T);
    else
      throw L1RpcException("unknown pattern type: " + pt);
      //edm::LogError("RPCTrigger") << "unknown pattern type: " + pt;

    CurPattern.SetRefGroup(rpcconst.StringToInt(XMLCh2String(attrs.getValue(Char2XMLCh("grp")))));
    CurPattern.SetQualityTabNumber(rpcconst.StringToInt(XMLCh2String(attrs.getValue(Char2XMLCh("qual")))));
    
    CurPattern.SetSign(rpcconst.StringToInt(XMLCh2String(attrs.getValue(Char2XMLCh("sign")))));
    CurPattern.SetCode(rpcconst.StringToInt(XMLCh2String(attrs.getValue(Char2XMLCh("code")))));
    CurPattern.SetNumber(rpcconst.StringToInt(XMLCh2String(attrs.getValue(Char2XMLCh("num")))));
  }
  else if (CurrElement == "str") {
    //<logstrip plane="LOGPLANE1" from="32" to="32"/>
    int logPlane = rpcconst.StringToInt(XMLCh2String(attrs.getValue(Char2XMLCh("Pl"))));
    CurPattern.SetStripFrom(logPlane, rpcconst.StringToInt(XMLCh2String(attrs.getValue(Char2XMLCh("f")))));
    CurPattern.SetStripTo(logPlane, rpcconst.StringToInt(XMLCh2String(attrs.getValue(Char2XMLCh("t")))) + 1);
  }
}

void L1RpcPatternsParser::endElement(const XMLCh* const uri, const XMLCh* const localname, const XMLCh* const qname) {
  string element = XMLCh2String(localname);
  if(element == "pat") {
    CurPacIt->second.push_back(CurPattern);
  }
}

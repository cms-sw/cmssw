/*******************************************************************************
*                                                                              *
*  Karol Bunkowski                                                             *
*  Warsaw University 2005                                                      *
*                                                                              *
*******************************************************************************/
#ifndef L1RpcPatternsParserH
#define L1RpcPatternsParserH
#include <string>
#include <iostream>

#include <xercesc/sax2/Attributes.hpp>
#include <xercesc/sax2/DefaultHandler.hpp>

//#include "L1Trigger/RPCTrigger/src/L1RpcParametersDef.h"
#include "L1Trigger/RPCTrigger/src/L1RpcParameters.h"
#include "L1Trigger/RPCTrigger/src/L1RpcPattern.h"
//#include "L1Trigger/RPCTrigger/src/L1RpcException.h"

XERCES_CPP_NAMESPACE_USE
class L1RpcPatternsParser : public DefaultHandler  {
public:
  //class SAX2PatHandler : public DefaultHandler {
  //public:
  //  SAX2PatHandler();
  //  ~SAX2PatHandler();

  //  void startElement(const XMLCh* const uri, const XMLCh* const localname, const XMLCh* const qname, const Attributes& attrs);
  //  void endElement (const XMLCh* const uri, const XMLCh* const localname, const XMLCh* const qname);
  //};

  L1RpcPatternsParser();
  ~L1RpcPatternsParser();

  
  void startElement(const XMLCh* const uri, const XMLCh* const localname, const XMLCh* const qname, const Attributes& attrs);
  void endElement (const XMLCh* const uri, const XMLCh* const localname, const XMLCh* const qname);

  void Parse(std::string fileName);

  const L1RpcPatternsVec& GetPatternsVec(const RPCParam::L1RpcConeCrdnts& coneCrds) const;

  struct TQuality {
    int QualityTabNumber;
    std::string FiredPlanes;
    short QualityValue;
  };

  typedef std::vector<TQuality> TQualityVec;

  const TQualityVec & GetQualityVec() const{ //XXX - clean me!
    return QualityVec;
  };


private:
  //virtual void startElement(const XMLCh* const name, xercesc::AttributeList& attributes);

  //virtual void endElement(const XMLCh* const name);

  static int InstanceCount;

  std::string CurrElement;

  TQualityVec QualityVec;

  typedef std::map<RPCParam::L1RpcConeCrdnts, L1RpcPatternsVec> TPatternsVecsMap;

  TPatternsVecsMap PatternsVecsMap;
  
  TPatternsVecsMap::iterator CurPacIt;

  L1RpcPattern CurPattern;
};

#endif

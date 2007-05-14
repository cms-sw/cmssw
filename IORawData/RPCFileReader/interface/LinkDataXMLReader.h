#ifndef LinkDataXMLReader_h
#define LinkDataXMLReader_h
/* This work is heavly based on KB code (L1RpcPatternXMLReader)*/

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Sources/interface/ExternalInputSource.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "IORawData/RPCFileReader/interface/RPCPacData.h"
#include "EventFilter/RPCRawToDigi/interface/EventRecords.h"

#include <xercesc/sax2/Attributes.hpp>
#include <xercesc/sax2/DefaultHandler.hpp>
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/framework/LocalFileFormatTarget.hpp>
#include <xercesc/framework/XMLPScanToken.hpp>
#include <xercesc/sax2/SAX2XMLReader.hpp>
#include "boost/bind.hpp"

#include <vector>
#include <string>
#include <map>

class FEDRawData;

XERCES_CPP_NAMESPACE_USE
/*############################################################################
#
#
#
############################################################################*/
class LinkDataXMLReader: public edm::ExternalInputSource, DefaultHandler 
{

   public:
  explicit LinkDataXMLReader(const edm::ParameterSet& iConfig,
			     edm::InputSourceDescription const& desc);
  virtual ~LinkDataXMLReader();

  virtual bool produce(edm::Event &);
  virtual void setRunAndEventInfo();


 private:

  FEDRawData * rawData(int fedId,  const std::vector<rpcrawtodigi::EventRecords> & result);
  std::pair<int,int> getDCCInputChannelNum(int tcNumber, int tbNumber);

  void startElement(const XMLCh* const uri, const XMLCh* const localname, 
                    const XMLCh* const qname, const Attributes& attrs);
  
  void endElement (const XMLCh* const uri, const XMLCh* const localname, const XMLCh* const qname);

  void clear();

  std::string IntToString(int i, int opt=0);

  int stringToInt(std::string str, int opt=0);

  int opticalLinkNum;
  int triggerCrateNum;
  int triggerBoardNum;
  bool isOpen_, noMoreData_, endOfEvent_;	
  bool endOfFile_;
  int eventPos_[2];
  int fileCounter_, eventCounter_;
  int run_, event_;
  std::string m_xmlDir;
  std::string m_CurrElement;
  static int m_instanceCount;

  std::map<int, std::vector<rpcrawtodigi::EventRecords> > results;

  XMLPScanToken aToken;
  SAX2XMLReader* parser;

};


#endif

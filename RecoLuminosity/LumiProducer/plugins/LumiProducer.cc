// -*- C++ -*-
//
// Package:    LumiProducer
// Class:      LumiProducer
// 
/**\class LumiProducer LumiProducer.cc RecoLuminosity/LumiProducer/src/LumiProducer.cc

Description: This class would load the luminosity object into a Luminosity Block

Implementation:
The are two main steps, the first one retrieve the record of the luminosity
data from the DB and the second loads the Luminosity Obj into the Lumi Block.
(Actually in the initial implementation it is retrieving from the ParameterSet
from the configuration file, the DB is not implemented yet)
*/
//
// Original Author:  Valerie Halyo
//                   David Dagenhart
//       
//         Created:  Tue Jun 12 00:47:28 CEST 2007
// $Id: LumiProducer.cc,v 1.10 2010/07/12 17:37:59 xiezhen Exp $

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "DataFormats/Provenance/interface/BranchType.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Luminosity/interface/LumiSummary.h"
#include "DataFormats/Luminosity/interface/LumiDetails.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "CoralBase/Exception.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/AttributeSpecification.h"
#include "CoralBase/Exception.h"
#include "CoralBase/Blob.h"
#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/ITransaction.h"
#include "RelationalAccess/AccessMode.h"
#include "RelationalAccess/ITypeConverter.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ICursor.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ITable.h"

#include "RecoLuminosity/LumiProducer/interface/DBService.h"
#include "RecoLuminosity/LumiProducer/interface/LumiNames.h"
#include "RecoLuminosity/LumiProducer/interface/Exception.h"
#include "RecoLuminosity/LumiProducer/interface/ConstantDef.h"
#include <sstream>
#include <string>
#include <memory>
#include <algorithm>
#include <vector>
#include <cstring>

#include <xercesc/dom/DOM.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/util/XMLString.hpp>

#include "boost/filesystem/path.hpp"
#include "boost/filesystem/operations.hpp"
namespace edm {
  class EventSetup;
}

//
// class declaration
//
class LumiProducer : public edm::EDProducer {

public:
  struct LSIdentifier{
    unsigned int runnumber;
    unsigned int lsnumber;
  };
  struct HLTData{
    unsigned int pathnum;
    unsigned int prescale;
    unsigned int l1passcount;
    unsigned int acceptcount;
  };
  struct L1Data{
    unsigned int bitnum;
    unsigned int prescale;
    unsigned int ratecount;
  };
  struct PerRunData{
    std::map< unsigned int,std::string > TRGBitNames;
    std::map< unsigned int,std::string > HLTPathNames;
  };
  struct PerLSData{
    float lumivalue;
    float lumierror;
    short lumiquality;
    unsigned long long deadcount;
    unsigned int numorbit;
    unsigned int startorbit;
    std::vector< HLTData > hltdata;
    std::vector< L1Data > l1data;
    std::vector< std::pair<LumiDetails::Algos, std::vector<float> > > bunchlumivalue;
    std::vector< std::pair<LumiDetails::Algos, std::vector<float> > > bunchlumierror;
    std::vector< std::pair<LumiDetails::Algos, std::vector<short> > > bunchlumiquality;
    std::vector< float > beam1intensity;
    std::vector< float > beam2intensity;
  };

  explicit LumiProducer(const edm::ParameterSet&);

  ~LumiProducer();
  
private:
  
  virtual void produce(edm::Event&, const edm::EventSetup&);

  virtual void beginRun(edm::Run&, edm::EventSetup const &);

  virtual void beginLuminosityBlock(edm::LuminosityBlock & iLBlock,
				    edm::EventSetup const& iSetup);
  virtual void endLuminosityBlock(edm::LuminosityBlock& lumiBlock, 
				  edm::EventSetup const& c);
  //void fillDefaultLumi(edm::LuminosityBlock & iLBlock);
  bool fillLumi(edm::LuminosityBlock & iLBlock);
  void fillRunCache(unsigned int runnumber);
  void fillLSCache(unsigned int luminum,unsigned int cachesize);
  void writeProductsForEntry(edm::LuminosityBlock & iLBlock,unsigned int runnumber,unsigned int luminum);

  //edm::ParameterSet pset_;

  std::string m_connectStr;
  std::string m_lumiversion;
  std::string m_siteconfpath;
  unsigned int m_cachedrun;
  PerRunData m_runcache;
  std::vector< unsigned int > m_cachedls;
  std::vector< PerLSData > m_lscache;
  bool m_isNullRun;
  
  const std::string servletTranslation(const std::string& servlet) const;
  std::string x2s(const XMLCh* input)const;
  XMLCh* s2x(const std::string& input)const;
  std::string toParentString(const xercesc::DOMNode &nodeToConvert)const;

};

//
// constructors and destructor
//

std::string 
LumiProducer::x2s(const XMLCh *toTranscode)const{
  std::string tmp(xercesc::XMLString::transcode(toTranscode));
  return tmp;
}

XMLCh*  
LumiProducer::s2x( const std::string& temp )const{
  XMLCh* buff = xercesc::XMLString::transcode(temp.c_str());    
  return  buff;
}

std::string
LumiProducer::toParentString(const xercesc::DOMNode &nodeToConvert)const{
  std::ostringstream oss;
  xercesc::DOMNodeList *childList = nodeToConvert.getChildNodes();

  unsigned int numNodes = childList->getLength ();
  for (unsigned int i = 0; i < numNodes; ++i){
    xercesc::DOMNode *childNode = childList->item(i);
    if (childNode->getNodeType() != xercesc::DOMNode::ELEMENT_NODE)
      continue;
    xercesc::DOMElement *child = static_cast < xercesc::DOMElement *> (childNode);
    xercesc::DOMNamedNodeMap *attributes = child->getAttributes();
    unsigned int numAttributes = attributes->getLength ();
    for (unsigned int j = 0; j < numAttributes; ++j){
      xercesc::DOMNode *attributeNode = attributes->item(j);
      if (attributeNode->getNodeType() != xercesc::DOMNode::ATTRIBUTE_NODE)
	continue;
      xercesc::DOMAttr *attribute = static_cast <xercesc::DOMAttr *> (attributeNode);
      
      oss << "(" << x2s(child->getTagName()) << 
	x2s(attribute->getName()) << "=" << 
	x2s(attribute->getValue()) << ")";
    }
  }
  return oss.str();
}

const std::string
LumiProducer::servletTranslation(const std::string& servlet) const{
  std::string frontierConnect;
  std::string realconnect;
  xercesc::XMLPlatformUtils::Initialize();  
  std::auto_ptr< xercesc::XercesDOMParser > parser(new xercesc::XercesDOMParser);
  try{
    parser->setValidationScheme(xercesc::XercesDOMParser::Val_Auto);
    parser->setDoNamespaces(false);
    parser->parse(m_siteconfpath.c_str());
    xercesc::DOMDocument* doc=parser->getDocument();
    if(!doc){
      return "";
    }
   
    xercesc::DOMNodeList *frontierConnectList=doc->getElementsByTagName(s2x("frontier-connect"));
    if (frontierConnectList->getLength()>0){
      xercesc::DOMElement *frontierConnectElement=static_cast < xercesc::DOMElement *> (frontierConnectList->item (0));
      frontierConnect = toParentString(*frontierConnectElement);
    }
    // Replace the last component of every "serverurl=" piece (up to the
    //   next close-paren) with the servlet
    std::string::size_type nextparen = 0;
    std::string::size_type serverurl, lastslash;
    std::string complexstr = "";
    while ((serverurl=frontierConnect.find("(serverurl=", nextparen)) != std::string::npos){
      realconnect.append(frontierConnect, nextparen, serverurl - nextparen);
      nextparen=frontierConnect.find(')', serverurl);
      lastslash=frontierConnect.rfind('/', nextparen);
      realconnect.append(frontierConnect,serverurl,lastslash-serverurl+1);
      realconnect.append(servlet);
    }
    realconnect.append(frontierConnect, nextparen,frontierConnect.length()-nextparen);
  }catch(xercesc::DOMException &e){
  }
  return realconnect;
}

LumiProducer::
LumiProducer::LumiProducer(const edm::ParameterSet& iConfig):m_cachedrun(0),m_isNullRun(false)
{
  // register your products
  produces<LumiSummary, edm::InLumi>();
  produces<LumiDetails, edm::InLumi>();
  // set up cache
  std::string connectStr=iConfig.getParameter<std::string>("connect");
  unsigned int cacheSize=iConfig.getUntrackedParameter<unsigned int>("ncacheEntries",5);
  m_cachedls.reserve(cacheSize);
  m_lscache.reserve(cacheSize);
  //m_runcache.TRGBitNames.reserve(200);
  //m_runcache.HLTPathNames.reserve(250);
  
  const std::string fproto("frontier://");
  //test if need frontier servlet site-local translation  
  if(connectStr.substr(0,fproto.length())==fproto){
    std::string::size_type startservlet=fproto.length();
    std::string::size_type endservlet=connectStr.find("(",startservlet);
    if(endservlet==std::string::npos){
      endservlet=connectStr.rfind('/',connectStr.length());
    }
    std::string servlet=connectStr.substr(startservlet,endservlet-startservlet);
    if( (servlet !="")&& (servlet.find_first_of(":/)[]")==std::string::npos)){
      if(servlet=="cms_conditions_data") servlet="";
      
      std::string siteconfpath=iConfig.getUntrackedParameter<std::string>("siteconfpath","");
      if(siteconfpath.length()==0){
	std::string url=(boost::filesystem::path("SITECONF")/boost::filesystem::path("local")/boost::filesystem::path("JobConfig")/boost::filesystem::path("site-local-config.xml")).string();
	char * tmp = getenv ("CMS_PATH");
	if(tmp){
	  m_siteconfpath = (boost::filesystem::path(tmp)/boost::filesystem::path(url)).string();
	}
      }else{
	if(!boost::filesystem::exists(boost::filesystem::path(siteconfpath))){
	  throw cms::Exception("Non existing path ")<<siteconfpath;
	}
	m_siteconfpath = (boost::filesystem::path(siteconfpath)/boost::filesystem::path("site-local-config.xml")).string();
      }
      //std::cout<<"servlet : "<<servlet<<std::endl;
      m_connectStr=fproto+servletTranslation(servlet)+connectStr.substr(endservlet);
    }else{
      m_connectStr=connectStr;
    }
  }else{
    m_connectStr=connectStr;
  }
  //std::cout<<"connect string "<< m_connectStr<<std::endl;
  m_lumiversion=iConfig.getParameter<std::string>("lumiversion");
}

LumiProducer::~LumiProducer(){ 
}
//
// member functions
//
void LumiProducer::produce(edm::Event& e, const edm::EventSetup& iSetup){ 
}
void LumiProducer::beginRun(edm::Run& run,edm::EventSetup const &iSetup){
  unsigned int runnumber=run.run();
  if( m_cachedrun==0 ){
    //no cache, fill m_runcache
    fillRunCache(runnumber);
  }else{
    //if there's cache
    //if runnumber match,read from cache, noaction here
    //if runnumber does not match,refresh runcache from DB
    if (m_cachedrun!=runnumber){
      m_cachedrun=runnumber;
      fillRunCache(runnumber);
    }
  }
}
void LumiProducer::beginLuminosityBlock(edm::LuminosityBlock &iLBlock, edm::EventSetup const &iSetup){
  
}
void LumiProducer::endLuminosityBlock(edm::LuminosityBlock & iLBlock, edm::EventSetup const& iSetup){
  unsigned int runnumber=iLBlock.run();
  unsigned int luminum=iLBlock.luminosityBlock();
  //if is null run, fill empty values and return
  if(m_isNullRun){
    std::auto_ptr<LumiSummary> pOut1;
    std::auto_ptr<LumiDetails> pOut2;
    LumiSummary* pIn1=new LumiSummary;
    LumiDetails* pIn2=new LumiDetails;
    pOut1.reset(pIn1);
    iLBlock.put(pOut1);
    pOut2.reset(pIn2);
    iLBlock.put(pOut2);
    return;
  }
  if (runnumber==m_cachedrun){
    //if runnumber and this LS is cached, read from cache
    if(std::find(m_cachedls.begin(),m_cachedls.end(),runnumber)==m_cachedls.end()){
      //if runnumber is cached but LS is not, this is the first LS, fill LS cache to full capacity
      unsigned int cachesize=m_cachedls.capacity();
      fillLSCache(luminum,cachesize);
    }
  }else{
    //this run is not cached. It shouldn't happen anyway.
    fillRunCache(runnumber);
    unsigned int cachesize=m_cachedls.capacity();
    fillLSCache(luminum,cachesize);
  }
  writeProductsForEntry(iLBlock,runnumber,luminum); 
}
void 
LumiProducer::fillRunCache(unsigned int runnumber){
}
void
LumiProducer::fillLSCache(unsigned int luminum,unsigned int cachesize){
}
void
LumiProducer::writeProductsForEntry(edm::LuminosityBlock & iLBlock,unsigned int runnumber,unsigned int luminum){
  std::vector<unsigned int>::iterator lspos=std::find(m_cachedls.begin(),m_cachedls.end(),luminum);
  size_t idx=std::distance(m_cachedls.begin(),lspos);
  PerLSData& lsdata=m_lscache[idx];
  std::auto_ptr<LumiSummary> pOut1;
  std::auto_ptr<LumiDetails> pOut2;
  LumiSummary* pIn1=new LumiSummary;
  LumiDetails* pIn2=new LumiDetails;
  pIn1->setLumiVersion(m_lumiversion);
  pIn1->setLumiData(lsdata.lumivalue,lsdata.lumierror,lsdata.lumiquality);
  pIn1->setDeadtime(lsdata.deadcount);
  pIn1->setOrbitData(lsdata.startorbit,lsdata.numorbit);
  std::vector<LumiSummary::L1> l1temp;
  for(std::vector< L1Data >::iterator it=lsdata.l1data.begin();it!=lsdata.l1data.end();++it){
    LumiSummary::L1 trgtmp;
    trgtmp.prescale=it->prescale;
    trgtmp.ratecount=it->ratecount;
    trgtmp.triggername=m_runcache.TRGBitNames[it->bitnum];
    l1temp.push_back(trgtmp);
  }
  std::vector<LumiSummary::HLT> hlttemp;
  for(std::vector< HLTData >::iterator it=lsdata.hltdata.begin();it!=lsdata.hltdata.end();++it){
    LumiSummary::HLT hlttmp;
    hlttmp.prescale=it->prescale;
    hlttmp.ratecount=it->acceptcount;
    hlttmp.inputcount=it->l1passcount;
    hlttmp.pathname=m_runcache.HLTPathNames[it->pathnum];
    hlttemp.push_back(hlttmp);
  }
  pIn1->swapL1Data(l1temp);
  pIn1->swapHLTData(hlttemp);
  for(unsigned int i=0;i<lsdata.bunchlumivalue.size();++i){
    pIn2->fill(lsdata.bunchlumivalue[i].first,lsdata.bunchlumivalue[i].second,lsdata.bunchlumierror[i].second,lsdata.bunchlumiquality[i].second);
  }
  pIn2->fillBeamIntensities(lsdata.beam1intensity,lsdata.beam2intensity);
  pOut1.reset(pIn1);
  iLBlock.put(pOut1);
  pOut2.reset(pIn2);
  iLBlock.put(pOut2);
}
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(LumiProducer);

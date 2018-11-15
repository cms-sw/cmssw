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
//                   Zhen Xie
//         Created:  Tue Jun 12 00:47:28 CEST 2007

#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "DataFormats/Provenance/interface/BranchType.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Luminosity/interface/LumiSummaryRunHeader.h"
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

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoLuminosity/LumiProducer/interface/DBService.h"
#include "RecoLuminosity/LumiProducer/interface/LumiNames.h"
#include "RecoLuminosity/LumiProducer/interface/Exception.h"
#include "RecoLuminosity/LumiProducer/interface/ConstantDef.h"
#include <iostream>
#include <sstream>
#include <string>
#include <memory>
#include <algorithm>
#include <vector>
#include <cstring>
#include <iterator>
#include <boost/foreach.hpp>
#include <boost/tokenizer.hpp>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include "FWCore/Concurrency/interface/Xerces.h"
#include <xercesc/util/XMLString.hpp>

#include "boost/filesystem/path.hpp"
#include "boost/filesystem/operations.hpp"

namespace edm {
  class EventSetup;
}

//
// class declaration
//
class LumiProducer : public edm::one::EDProducer<edm::one::WatchRuns,
                                                 edm::BeginLuminosityBlockProducer,
                                                 edm::EndRunProducer> {

public:

  struct HLTData{
    std::string pathname;
    unsigned int prescale;
    unsigned int l1passcount;
    unsigned int acceptcount;
  };
  struct L1Data{
    std::string bitname;
    unsigned int prescale;
    unsigned int ratecount;
  };
  struct PerRunData{
    std::string bitzeroname;//norm bit name
    std::map<std::string, unsigned int> TRGBitNameToIndex;
    std::map<std::string, unsigned int> HLTPathNameToIndex;
    std::vector<std::string>            TRGBitNames;
    std::vector<std::string>            HLTPathNames;
  };
  struct PerLSData{
    float lumivalue;
    float lumierror;
    short lumiquality;
    unsigned long long deadcount;
    unsigned int numorbit;
    unsigned int startorbit;
    unsigned int bitzerocount;
    unsigned int bitzeroprescale;
    std::vector< HLTData > hltdata;
    std::vector< L1Data > l1data;
    std::vector< std::pair<std::string, std::vector<float> > > bunchlumivalue;
    std::vector< std::pair<std::string, std::vector<float> > > bunchlumierror;
    std::vector< std::pair<std::string, std::vector<short> > > bunchlumiquality;
    std::vector<float> beam1intensity;
    std::vector<float> beam2intensity;
  };

  explicit LumiProducer(const edm::ParameterSet&);

  ~LumiProducer() override;
  
private:
  
  void produce(edm::Event&, const edm::EventSetup&) final;

  void beginRun(edm::Run const&, edm::EventSetup const &) final;

  void beginLuminosityBlockProduce(edm::LuminosityBlock & iLBlock,
				    edm::EventSetup const& iSetup) final;
 
  void endRun(edm::Run const&, edm::EventSetup const &) final;
  void endRunProduce(edm::Run&, edm::EventSetup const &) final;

  bool fillLumi(edm::LuminosityBlock & iLBlock);
  void fillRunCache(const coral::ISchema& schema,unsigned int runnumber);
  void fillLSCache(unsigned int luminum);
  void writeProductsForEntry(edm::LuminosityBlock & iLBlock,unsigned int runnumber,unsigned int luminum);
  const std::string servletTranslation(const std::string& servlet) const;
  std::string x2s(const XMLCh* input)const;
  XMLCh* s2x(const std::string& input)const;
  std::string toParentString(const xercesc::DOMNode &nodeToConvert)const;
  unsigned long long getLumiDataId(const coral::ISchema& schema,unsigned int runnumber);
  unsigned long long getTrgDataId(const coral::ISchema& schema,unsigned int runnumber);
  unsigned long long getHltDataId(const coral::ISchema& schema,unsigned int runnumber);
  std::string getCurrentDataTag(const coral::ISchema& schema);
  std::string m_connectStr;
  std::string m_lumiversion;
  std::string m_siteconfpath;
  unsigned int m_cachedrun;
  unsigned long long m_cachedlumidataid;
  unsigned long long m_cachedtrgdataid;
  unsigned long long m_cachedhltdataid;
  PerRunData  m_runcache;
  std::map< unsigned int,PerLSData > m_lscache;
  bool m_isNullRun;
  unsigned int m_cachesize;
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
  cms::concurrency::xercesInitialize();  
  std::unique_ptr< xercesc::XercesDOMParser > parser(new xercesc::XercesDOMParser);
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
LumiProducer::LumiProducer(const edm::ParameterSet& iConfig):m_cachedrun(0),m_isNullRun(false),m_cachesize(0)
{
  // register your products
  produces<LumiSummaryRunHeader, edm::Transition::EndRun>();
  produces<LumiSummary, edm::Transition::BeginLuminosityBlock>();
  produces<LumiDetails, edm::Transition::BeginLuminosityBlock>();
  // set up cache
  std::string connectStr=iConfig.getParameter<std::string>("connect");
  m_cachesize=iConfig.getUntrackedParameter<unsigned int>("ncacheEntries",5);
  m_lumiversion=iConfig.getUntrackedParameter<std::string>("lumiversion","");
  const std::string fproto("frontier://");
  //test if need frontier servlet site-local translation  
  if(connectStr.substr(0,fproto.length())==fproto){
    std::string::size_type startservlet=fproto.length();
    std::string::size_type endservlet=connectStr.find("(",startservlet);
    if(endservlet==std::string::npos){
      endservlet=connectStr.rfind('/',connectStr.length());
    }
    std::string servlet=connectStr.substr(startservlet,endservlet-startservlet);
    if( (!servlet.empty())&& (servlet.find_first_of(":/)[]")==std::string::npos)){
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
}

LumiProducer::~LumiProducer(){ 
}

//
// member functions
//
void LumiProducer::produce(edm::Event& e, const edm::EventSetup& iSetup)
{ 
}
unsigned long long 
LumiProducer::getLumiDataId(const coral::ISchema& schema,unsigned int runnumber){ 
  //
  //select max(data_id) from lumidata where runnum=:runnum
  //
  //std::count<<"entering getLumiDataId "<<std::endl;
  unsigned long long lumidataid=0;
  coral::AttributeList bindVariables;
  bindVariables.extend("runnum",typeid(unsigned int));
  bindVariables["runnum"].data<unsigned int>()=runnumber;
  coral::AttributeList lumiidOutput;
  lumiidOutput.extend("lumidataid",typeid(unsigned long long));
  coral::IQuery* lumiQuery=schema.newQuery();
  lumiQuery->addToTableList(lumi::LumiNames::lumidataTableName());
  lumiQuery->addToOutputList("MAX(DATA_ID)","lumidataid");
  lumiQuery->setCondition("RUNNUM=:runnum",bindVariables);
  lumiQuery->defineOutput(lumiidOutput);
  coral::ICursor& lumicursor=lumiQuery->execute();
  while( lumicursor.next() ){
    const coral::AttributeList& row=lumicursor.currentRow();
    if(!row["lumidataid"].isNull()){
      lumidataid=row["lumidataid"].data<unsigned long long>();
    }
  }
  delete lumiQuery;
  return lumidataid;
}
unsigned long long 
LumiProducer::getTrgDataId(const coral::ISchema& schema,unsigned int runnumber){
  //
  //select max(data_id) from trgdata where runnum=:runnum
  //
  unsigned long long trgdataid=0;
  coral::AttributeList bindVariables;
  bindVariables.extend("runnum",typeid(unsigned int));
  bindVariables["runnum"].data<unsigned int>()=runnumber;
  coral::AttributeList trgidOutput;
  trgidOutput.extend("trgdataid",typeid(unsigned long long));
  coral::IQuery* trgQuery=schema.newQuery();
  trgQuery->addToTableList(lumi::LumiNames::trgdataTableName());
  trgQuery->addToOutputList("MAX(DATA_ID)","trgdataid");
  trgQuery->setCondition("RUNNUM=:runnum",bindVariables);
  trgQuery->defineOutput(trgidOutput);
  coral::ICursor& trgcursor=trgQuery->execute();
  while( trgcursor.next() ){
    const coral::AttributeList& row=trgcursor.currentRow();
    if(!row["trgdataid"].isNull()){
      trgdataid=row["trgdataid"].data<unsigned long long>();
    }
  }
  delete trgQuery;
  return trgdataid;
}
unsigned long long 
LumiProducer::getHltDataId(const coral::ISchema& schema,unsigned int runnumber){
  //
  //select max(data_id) from hltdata where runnum=:runnum
  //
  unsigned long long hltdataid=0;
  coral::AttributeList bindVariables;
  bindVariables.extend("runnum",typeid(unsigned int));
  bindVariables["runnum"].data<unsigned int>()=runnumber;
  coral::AttributeList hltidOutput;
  hltidOutput.extend("hltdataid",typeid(unsigned long long));
  coral::IQuery* hltQuery=schema.newQuery();
  hltQuery->addToTableList(lumi::LumiNames::hltdataTableName());
  hltQuery->addToOutputList("MAX(DATA_ID)","hltdataid");
  hltQuery->setCondition("RUNNUM=:runnum",bindVariables);
  hltQuery->defineOutput(hltidOutput);
  coral::ICursor& hltcursor=hltQuery->execute();
  while( hltcursor.next() ){
    const coral::AttributeList& row=hltcursor.currentRow();
    if(!row["hltdataid"].isNull()){
      hltdataid=row["hltdataid"].data<unsigned long long>();
    }
  }
  delete hltQuery;
  return hltdataid;
}

std::string 
LumiProducer::getCurrentDataTag(const coral::ISchema& schema){
  //select tagid,tagname from tags
  std::string result;
  std::map<unsigned long long,std::string> alltags;
  coral::IQuery* tagQuery=schema.newQuery();
  tagQuery->addToTableList(lumi::LumiNames::tagsTableName());
  tagQuery->addToOutputList("TAGID");
  tagQuery->addToOutputList("TAGNAME");
  coral::AttributeList tagoutput;
  tagoutput.extend("TAGID",typeid(unsigned long long));
  tagoutput.extend("TAGNAME",typeid(std::string));
  tagQuery->defineOutput(tagoutput);
  coral::ICursor& tagcursor=tagQuery->execute();
  while( tagcursor.next() ){
    const coral::AttributeList& row=tagcursor.currentRow();
    unsigned long long tagid=row["TAGID"].data<unsigned long long>();
    const std::string  tagname=row["TAGNAME"].data<std::string>();
    alltags.insert(std::make_pair(tagid,tagname));
  }
  delete tagQuery;
  unsigned long long maxid=0;
  for(std::map<unsigned long long,std::string>::iterator it = alltags.begin(); it !=alltags.end(); ++it) {
    if( it->first > maxid){
      maxid=it->first;
    }
  }
  result=alltags[maxid];
  return result;
}

void 
LumiProducer::beginRun(edm::Run const& run,edm::EventSetup const &iSetup)
{
  unsigned int runnumber=run.run();
  if(m_cachedrun!=runnumber){
    //queries once per run
    m_cachedrun=runnumber;
    edm::Service<lumi::service::DBService> mydbservice;
    if( !mydbservice.isAvailable() ){
      throw cms::Exception("Non existing service lumi::service::DBService");
    }
    auto session=mydbservice->connectReadOnly(m_connectStr);
    try{
      session->transaction().start(true);
      m_cachedlumidataid=getLumiDataId(session->nominalSchema(),runnumber);
      if(m_cachedlumidataid!=0){//if no lumi, do not bother other info
	m_cachedtrgdataid=getTrgDataId(session->nominalSchema(),runnumber);
	m_cachedhltdataid=getHltDataId(session->nominalSchema(),runnumber);
	fillRunCache(session->nominalSchema(),runnumber);
      }else{
	m_isNullRun=true;
      }
      session->transaction().commit();
    }catch(const coral::Exception& er){
      session->transaction().rollback();
      throw cms::Exception("DatabaseError ")<<er.what();
    }
  }
  //std::cout<<"end of beginRun "<<runnumber<<std::endl;
}

void LumiProducer::beginLuminosityBlockProduce(edm::LuminosityBlock &iLBlock, edm::EventSetup const &iSetup)
{
  unsigned int runnumber=iLBlock.run();
  unsigned int luminum=iLBlock.luminosityBlock();
  //std::cout<<"beg of beginLuminosityBlock "<<luminum<<std::endl;
  //if is null run, fill empty values and return
  if(m_isNullRun){
    iLBlock.put(std::make_unique<LumiSummary>());
    iLBlock.put(std::make_unique<LumiDetails>());
    return;
  }
  if(m_lscache.find(luminum)==m_lscache.end()){
    //if runnumber is cached but LS is not, this is the first LS, fill LS cache to full capacity
    fillLSCache(luminum);
  }
  //here the presence of ls is guaranteed
  writeProductsForEntry(iLBlock,runnumber,luminum); 
}
void 
LumiProducer::endRun(edm::Run const& run,edm::EventSetup const &iSetup)
{}
void 
LumiProducer::endRunProduce(edm::Run& run,edm::EventSetup const &iSetup)
{
  auto lsrh = std::make_unique<LumiSummaryRunHeader>();
  lsrh->swapL1Names(m_runcache.TRGBitNames);
  lsrh->swapHLTNames(m_runcache.HLTPathNames);
  run.put(std::move(lsrh));
  m_runcache.TRGBitNameToIndex.clear();
  m_runcache.HLTPathNameToIndex.clear();
}
void 
LumiProducer::fillRunCache(const coral::ISchema& schema,unsigned int runnumber){
  if(m_lumiversion.empty()){
    m_lumiversion=getCurrentDataTag(schema);
  }
  std::cout<<"lumi tag version 2 "<<m_lumiversion<<std::endl;
  if(m_cachedtrgdataid!=0){
    coral::AttributeList trgBindVariables;
    trgBindVariables.extend("trgdataid",typeid(unsigned long long));
    trgBindVariables["trgdataid"].data<unsigned long long>()=m_cachedtrgdataid;
    //std::cout<<"cached trgdataid "<<m_cachedtrgdataid<<std::endl;
    coral::AttributeList trgOutput;
    trgOutput.extend("bitzeroname",typeid(std::string));
    trgOutput.extend("bitnameclob",typeid(std::string));
    coral::IQuery* trgQuery=schema.newQuery();
    trgQuery->addToTableList(lumi::LumiNames::trgdataTableName());
    trgQuery->addToOutputList("BITZERONAME");
    trgQuery->addToOutputList("BITNAMECLOB");
    trgQuery->setCondition("DATA_ID=:trgdataid",trgBindVariables);
    trgQuery->defineOutput(trgOutput);
    coral::ICursor& trgcursor=trgQuery->execute();
    while( trgcursor.next() ){
      const coral::AttributeList& row=trgcursor.currentRow();
      m_runcache.bitzeroname=row["bitzeroname"].data<std::string>();
      //std::cout<<"bitzeroname "<<m_runcache.bitzeroname<<std::endl;
      std::string bitnames=row["bitnameclob"].data<std::string>();
      boost::char_separator<char> sep(",");
      boost::tokenizer<boost::char_separator<char> > tokens(bitnames,sep);
      for(boost::tokenizer<boost::char_separator<char> >::iterator tok_it=tokens.begin();tok_it!=tokens.end();++tok_it){
	m_runcache.TRGBitNames.push_back(*tok_it);
      }
      for(unsigned int i=0;i<m_runcache.TRGBitNames.size();++i){
	m_runcache.TRGBitNameToIndex.insert(std::make_pair(m_runcache.TRGBitNames.at(i),i) );
      }      
    }
    delete trgQuery;
  }
  if(m_cachedhltdataid!=0){
    //
    //select pathnameclob from hltdata where data_id=:hltdataid
    //
    coral::AttributeList hltBindVariables;
    hltBindVariables.extend("hltdataid",typeid(unsigned long long));
    hltBindVariables["hltdataid"].data<unsigned long long>()=m_cachedhltdataid;
    coral::AttributeList hltOutput;
    hltOutput.extend("PATHNAMECLOB",typeid(std::string));
    coral::IQuery* hltQuery=schema.newQuery();
    hltQuery->addToTableList(lumi::LumiNames::hltdataTableName());
    hltQuery->addToOutputList("PATHNAMECLOB");
    hltQuery->setCondition("DATA_ID=:hltdataid",hltBindVariables);
    hltQuery->defineOutput(hltOutput);
    coral::ICursor& hltcursor=hltQuery->execute();
    while( hltcursor.next() ){
      const coral::AttributeList& row=hltcursor.currentRow();
      std::string pathnames=row["PATHNAMECLOB"].data<std::string>();
      boost::char_separator<char> sep(",");
      boost::tokenizer<boost::char_separator<char> > tokens(pathnames,sep);
      for(boost::tokenizer<boost::char_separator<char> >::iterator tok_it=tokens.begin();tok_it!=tokens.end();++tok_it){
	m_runcache.HLTPathNames.push_back(*tok_it);
      }
      for(unsigned int i=0;i<m_runcache.HLTPathNames.size();++i){
	m_runcache.HLTPathNameToIndex.insert(std::make_pair(m_runcache.HLTPathNames.at(i),i));
      }     
    }
    delete hltQuery;   
  }
}
void
LumiProducer::fillLSCache(unsigned int luminum){
  //initialize cache
  if(m_isNullRun) return;
  m_lscache.clear();
  for(unsigned int n=luminum;n<luminum+m_cachesize;++n){
    PerLSData l;
    l.hltdata.reserve(250);
    l.l1data.reserve(192);
    l.bunchlumivalue.reserve(5);
    l.bunchlumierror.reserve(5);
    l.bunchlumiquality.reserve(5);
    l.beam1intensity.resize(3564,0.0);
    l.beam2intensity.resize(3564,0.0);
    m_lscache.insert(std::make_pair(n,l));
  }
  //queries once per cache refill
  //
  //select cmslsnum,instlumi,startorbit,numorbit,bxindex,beam1intensity,beam2intensity,bxlumivalue_occ1,bxlumivalue_occ2,bxlumivalue_et from lumisummaryv2 where cmslsnum>=:lsmin and cmslsnum<:lsmax and data_id=:lumidataid;
  //
  edm::Service<lumi::service::DBService> mydbservice;
  if( !mydbservice.isAvailable() ){
    throw cms::Exception("Non existing service lumi::service::DBService");
  }
  auto session=mydbservice->connectReadOnly(m_connectStr);
  try{
    session->transaction().start(true);
    coral::ISchema& schema=session->nominalSchema();
    coral::AttributeList lumisummaryBindVariables;
    lumisummaryBindVariables.extend("lsmin",typeid(unsigned int));
    lumisummaryBindVariables.extend("lsmax",typeid(unsigned int));
    lumisummaryBindVariables.extend("lumidataid",typeid(unsigned long long));
    lumisummaryBindVariables["lumidataid"].data<unsigned long long>()=m_cachedlumidataid;
    lumisummaryBindVariables["lsmin"].data<unsigned int>()=luminum;
    lumisummaryBindVariables["lsmax"].data<unsigned int>()=luminum+m_cachesize;
    coral::AttributeList lumisummaryOutput;
    lumisummaryOutput.extend("CMSLSNUM",typeid(unsigned int));
    lumisummaryOutput.extend("INSTLUMI",typeid(float));
    lumisummaryOutput.extend("STARTORBIT",typeid(unsigned int));
    lumisummaryOutput.extend("NUMORBIT",typeid(unsigned int));
    lumisummaryOutput.extend("CMSBXINDEXBLOB",typeid(coral::Blob));
    lumisummaryOutput.extend("BEAMINTENSITYBLOB_1",typeid(coral::Blob));
    lumisummaryOutput.extend("BEAMINTENSITYBLOB_2",typeid(coral::Blob));
    lumisummaryOutput.extend("BXLUMIVALUE_OCC1",typeid(coral::Blob));
    lumisummaryOutput.extend("BXLUMIVALUE_OCC2",typeid(coral::Blob));
    lumisummaryOutput.extend("BXLUMIVALUE_ET",typeid(coral::Blob));
    coral::IQuery* lumisummaryQuery=schema.newQuery();
    lumisummaryQuery->addToTableList(lumi::LumiNames::lumisummaryv2TableName());
    lumisummaryQuery->addToOutputList("CMSLSNUM");
    lumisummaryQuery->addToOutputList("INSTLUMI");
    lumisummaryQuery->addToOutputList("STARTORBIT");
    lumisummaryQuery->addToOutputList("NUMORBIT");
    lumisummaryQuery->addToOutputList("CMSBXINDEXBLOB");
    lumisummaryQuery->addToOutputList("BEAMINTENSITYBLOB_1");
    lumisummaryQuery->addToOutputList("BEAMINTENSITYBLOB_2");
    lumisummaryQuery->addToOutputList("BXLUMIVALUE_OCC1");
    lumisummaryQuery->addToOutputList("BXLUMIVALUE_OCC2");
    lumisummaryQuery->addToOutputList("BXLUMIVALUE_ET");
    lumisummaryQuery->setCondition("CMSLSNUM>=:lsmin AND CMSLSNUM<:lsmax AND DATA_ID=:lumidataid",lumisummaryBindVariables);
    lumisummaryQuery->defineOutput(lumisummaryOutput);
    coral::ICursor& lumisummarycursor=lumisummaryQuery->execute();
    unsigned int rowcounter=0;
    while( lumisummarycursor.next() ){
      const coral::AttributeList& row=lumisummarycursor.currentRow();
      unsigned int cmslsnum=row["CMSLSNUM"].data<unsigned int>();
      //std::cout<<"cmslsnum "<<cmslsnum<<std::endl;
      PerLSData& lsdata=m_lscache[cmslsnum];
      lsdata.lumivalue=row["INSTLUMI"].data<float>();
      lsdata.lumierror=0.0;
      lsdata.lumiquality=0;
      lsdata.startorbit=row["STARTORBIT"].data<unsigned int>();
      lsdata.numorbit=row["NUMORBIT"].data<unsigned int>();
      
      if(!row["CMSBXINDEXBLOB"].isNull() && !row["BXLUMIVALUE_OCC1"].isNull() ){
	const coral::Blob& bxindexBlob=row["CMSBXINDEXBLOB"].data<coral::Blob>();
	const void* bxindex_StartAddress=bxindexBlob.startingAddress();
	short* bxindex=(short*)::malloc(bxindexBlob.size());
	const coral::Blob& beam1intensityBlob=row["BEAMINTENSITYBLOB_1"].data<coral::Blob>();
	const void* beam1intensityBlob_StartAddress=beam1intensityBlob.startingAddress();
	float* beam1intensity=(float*)::malloc(beam1intensityBlob.size());
	const coral::Blob& beam2intensityBlob=row["BEAMINTENSITYBLOB_2"].data<coral::Blob>();
	const void* beam2intensityBlob_StartAddress=beam2intensityBlob.startingAddress();
	float* beam2intensity=(float*)::malloc(beam2intensityBlob.size());
	std::memmove(bxindex,bxindex_StartAddress,bxindexBlob.size());
	std::memmove(beam1intensity,beam1intensityBlob_StartAddress,beam1intensityBlob.size());
	std::memmove(beam2intensity,beam2intensityBlob_StartAddress,beam2intensityBlob.size());

	unsigned int iMax = bxindexBlob.size()/sizeof(short);
	unsigned int lsb1Max = lsdata.beam1intensity.size();
	unsigned int lsb2Max = lsdata.beam2intensity.size();
	unsigned int ib1Max = beam1intensityBlob.size()/sizeof(float);
	unsigned int ib2Max = beam2intensityBlob.size()/sizeof(float);
	for(unsigned int i=0;i<iMax;++i){
	  unsigned int idx=bxindex[i];
	  if(ib1Max>i && lsb1Max>idx){
	    lsdata.beam1intensity.at(idx)=beam1intensity[i];
	  }
	  if(ib2Max>i && lsb2Max>idx){
	    lsdata.beam2intensity.at(idx)=beam2intensity[i];
	  }
	}
	::free(bxindex);
	::free(beam1intensity);
	::free(beam2intensity);

	const coral::Blob& bxlumivalBlob_occ1=row["BXLUMIVALUE_OCC1"].data<coral::Blob>();
	const void* bxlumival_occ1_StartAddress=bxlumivalBlob_occ1.startingAddress();
	float* bxlumival_occ1=(float*)::malloc(bxlumivalBlob_occ1.size());
	std::memmove(bxlumival_occ1,bxlumival_occ1_StartAddress,bxlumivalBlob_occ1.size());
	std::vector<float> bxlumivalVec_occ1(bxlumival_occ1,bxlumival_occ1+bxlumivalBlob_occ1.size()/sizeof(float));
	::free(bxlumival_occ1);
	lsdata.bunchlumivalue.push_back(std::make_pair(std::string("OCC1"),bxlumivalVec_occ1));
	lsdata.bunchlumierror.push_back(std::make_pair(std::string("OCC1"),std::vector<float>(3564)));
	lsdata.bunchlumiquality.push_back(std::make_pair(std::string("OCC1"),std::vector<short>(3564)));
	const coral::Blob& bxlumivalBlob_occ2=row["BXLUMIVALUE_OCC2"].data<coral::Blob>();
	const void* bxlumival_occ2_StartAddress=bxlumivalBlob_occ2.startingAddress();
	float* bxlumival_occ2=(float*)::malloc(bxlumivalBlob_occ2.size());
	std::memmove(bxlumival_occ2,bxlumival_occ2_StartAddress,bxlumivalBlob_occ2.size());
	std::vector<float> bxlumivalVec_occ2(bxlumival_occ2,bxlumival_occ2+bxlumivalBlob_occ1.size()/sizeof(float));
	::free(bxlumival_occ2);
	lsdata.bunchlumivalue.push_back(std::make_pair(std::string("OCC2"),bxlumivalVec_occ2));
	lsdata.bunchlumierror.push_back(std::make_pair(std::string("OCC2"),std::vector<float>(3564)));
	lsdata.bunchlumiquality.push_back(std::make_pair(std::string("OCC2"),std::vector<short>(3564)));

	const coral::Blob& bxlumivalBlob_et=row["BXLUMIVALUE_ET"].data<coral::Blob>();
	const void* bxlumival_et_StartAddress=bxlumivalBlob_et.startingAddress();
	float* bxlumival_et=(float*)::malloc(bxlumivalBlob_et.size());
	std::memmove(bxlumival_et,bxlumival_et_StartAddress,bxlumivalBlob_et.size());	
	std::vector<float> bxlumivalVec_et(bxlumival_et,bxlumival_et+bxlumivalBlob_et.size()/sizeof(float));
	::free(bxlumival_et);
	lsdata.bunchlumivalue.push_back(std::make_pair(std::string("ET"),bxlumivalVec_et));
	lsdata.bunchlumierror.push_back(std::make_pair(std::string("ET"),std::vector<float>(3564)));
	lsdata.bunchlumiquality.push_back(std::make_pair(std::string("ET"),std::vector<short>(3564)));
      }
      ++rowcounter;
    }
    if (rowcounter==0){
      m_isNullRun=true;
      return;
    }
    delete lumisummaryQuery;
    
    //
    //select cmslsnum,deadtimecount,bitzerocount,bitzeroprescale,prescaleblob,trgcountblob from lstrg where cmslsnum >=:luminum and cmslsnum<:luminum+cachesize AND data_id=:trgdataid;
    //
    coral::AttributeList trgBindVariables;
    trgBindVariables.extend("lsmin",typeid(unsigned int));
    trgBindVariables.extend("lsmax",typeid(unsigned int));
    trgBindVariables.extend("trgdataid",typeid(unsigned long long));
    trgBindVariables["lsmin"].data<unsigned int>()=luminum;
    trgBindVariables["lsmax"].data<unsigned int>()=luminum+m_cachesize;
    trgBindVariables["trgdataid"].data<unsigned long long>()=m_cachedtrgdataid;
    coral::AttributeList trgOutput;
    trgOutput.extend("CMSLSNUM",typeid(unsigned int));
    trgOutput.extend("DEADTIMECOUNT",typeid(unsigned long long));
    trgOutput.extend("BITZEROCOUNT",typeid(unsigned int));
    trgOutput.extend("BITZEROPRESCALE",typeid(unsigned int));
    trgOutput.extend("PRESCALEBLOB",typeid(coral::Blob));
    trgOutput.extend("TRGCOUNTBLOB",typeid(coral::Blob));

    coral::IQuery* trgQuery=schema.newQuery();
    trgQuery->addToTableList(lumi::LumiNames::lstrgTableName());
    trgQuery->addToOutputList("CMSLSNUM");
    trgQuery->addToOutputList("DEADTIMECOUNT");
    trgQuery->addToOutputList("BITZEROCOUNT");
    trgQuery->addToOutputList("BITZEROPRESCALE");
    trgQuery->addToOutputList("PRESCALEBLOB");
    trgQuery->addToOutputList("TRGCOUNTBLOB");
    trgQuery->setCondition("CMSLSNUM>=:lsmin AND CMSLSNUM<:lsmax AND DATA_ID=:trgdataid",trgBindVariables);
    trgQuery->defineOutput(trgOutput);
    coral::ICursor& trgcursor=trgQuery->execute();
    while( trgcursor.next() ){
      const coral::AttributeList& row=trgcursor.currentRow();
      unsigned int cmslsnum=row["CMSLSNUM"].data<unsigned int>();
      PerLSData& lsdata=m_lscache[cmslsnum];
      lsdata.deadcount=row["DEADTIMECOUNT"].data<unsigned long long>();
      lsdata.bitzerocount=row["BITZEROCOUNT"].data<unsigned int>();
      lsdata.bitzeroprescale=row["BITZEROPRESCALE"].data<unsigned int>();
      if(!row["PRESCALEBLOB"].isNull()){
	const coral::Blob& prescaleblob=row["PRESCALEBLOB"].data<coral::Blob>();
	const void* prescaleblob_StartAddress=prescaleblob.startingAddress();
	unsigned int* prescales=(unsigned int*)::malloc(prescaleblob.size());
	std::memmove(prescales,prescaleblob_StartAddress,prescaleblob.size());
	const coral::Blob& trgcountblob=row["TRGCOUNTBLOB"].data<coral::Blob>();
	const void* trgcountblob_StartAddress=trgcountblob.startingAddress();
	unsigned int* trgcounts=(unsigned int*)::malloc(trgcountblob.size());
	std::memmove(trgcounts,trgcountblob_StartAddress,trgcountblob.size());
	for(unsigned int i=0; i < trgcountblob.size()/sizeof(unsigned int); ++i){
	  L1Data l1tmp;
	  l1tmp.bitname=m_runcache.TRGBitNames[i];
	  l1tmp.prescale=prescales[i];
	  l1tmp.ratecount=trgcounts[i];
	  lsdata.l1data.push_back(l1tmp);
	}
	::free(prescales);
	::free(trgcounts);
      }
    }
    delete trgQuery;
    //
    //select cmslsnum,hltcountblob,hltacceptblob,prescaleblob from hlt where cmslsnum >=:luminum and cmslsnum<=:luminum+cachesize and data_id=:hltdataid 
    //
    coral::AttributeList hltBindVariables;
    hltBindVariables.extend("lsmin",typeid(unsigned int));
    hltBindVariables.extend("lsmax",typeid(unsigned int));
    hltBindVariables.extend("hltdataid",typeid(unsigned long long));
    hltBindVariables["lsmin"].data<unsigned int>()=luminum;
    hltBindVariables["lsmax"].data<unsigned int>()=luminum+m_cachesize;
    hltBindVariables["hltdataid"].data<unsigned long long>()=m_cachedhltdataid;
    coral::AttributeList hltOutput;
    hltOutput.extend("CMSLSNUM",typeid(unsigned int));
    hltOutput.extend("HLTCOUNTBLOB",typeid(coral::Blob));
    hltOutput.extend("HLTACCEPTBLOB",typeid(coral::Blob));
    hltOutput.extend("PRESCALEBLOB",typeid(coral::Blob));
    coral::IQuery* hltQuery=schema.newQuery();
    hltQuery->addToTableList(lumi::LumiNames::lshltTableName());
    hltQuery->addToOutputList("CMSLSNUM");
    hltQuery->addToOutputList("HLTCOUNTBLOB");
    hltQuery->addToOutputList("HLTACCEPTBLOB");
    hltQuery->addToOutputList("PRESCALEBLOB");
    hltQuery->setCondition("CMSLSNUM>=:lsmin AND CMSLSNUM<:lsmax AND DATA_ID=:hltdataid",hltBindVariables);
    hltQuery->defineOutput(hltOutput);
    coral::ICursor& hltcursor=hltQuery->execute();
    while( hltcursor.next() ){
      const coral::AttributeList& row=hltcursor.currentRow();   
      unsigned int cmslsnum=row["CMSLSNUM"].data<unsigned int>();
      PerLSData& lsdata=m_lscache[cmslsnum];
      if(!row["PRESCALEBLOB"].isNull()){
	const coral::Blob& hltprescaleblob=row["PRESCALEBLOB"].data<coral::Blob>();
	const void* hltprescaleblob_StartAddress=hltprescaleblob.startingAddress();
	unsigned int* hltprescales=(unsigned int*)::malloc(hltprescaleblob.size());
	std::memmove(hltprescales,hltprescaleblob_StartAddress,hltprescaleblob.size());
	const coral::Blob& hltcountblob=row["HLTCOUNTBLOB"].data<coral::Blob>();
	const void* hltcountblob_StartAddress=hltcountblob.startingAddress();
	unsigned int* hltcounts=(unsigned int*)::malloc(hltcountblob.size());
	std::memmove(hltcounts,hltcountblob_StartAddress,hltcountblob.size());
	const coral::Blob& hltacceptblob=row["HLTACCEPTBLOB"].data<coral::Blob>();
	const void* hltacceptblob_StartAddress=hltacceptblob.startingAddress();
	unsigned int* hltaccepts=(unsigned int*)::malloc(hltacceptblob.size());
	std::memmove(hltaccepts,hltacceptblob_StartAddress,hltacceptblob.size()); 	
	unsigned int nhltaccepts = hltacceptblob.size()/sizeof(unsigned int);
        if(nhltaccepts > 0 && m_runcache.HLTPathNames.empty()){
          edm::LogWarning("CorruptOrMissingHLTData")<<"Got "<<nhltaccepts
<<" hltaccepts, but the run chache is empty. hltdata will  not be written";
            break;
        }

	for(unsigned int i=0; i < hltacceptblob.size()/sizeof(unsigned int); ++i){
	  HLTData hlttmp;
	  hlttmp.pathname=m_runcache.HLTPathNames[i];
	  hlttmp.prescale=hltprescales[i];
	  hlttmp.l1passcount=hltcounts[i];
	  hlttmp.acceptcount=hltaccepts[i];
	  lsdata.hltdata.push_back(hlttmp);
	}
	::free(hltprescales);
	::free(hltcounts);
	::free(hltaccepts);
      }
    }
    delete hltQuery;
    session->transaction().commit();
  }catch(const coral::Exception& er){
    session->transaction().rollback();
    throw cms::Exception("DatabaseError ")<<er.what();
  }
}
void
LumiProducer::writeProductsForEntry(edm::LuminosityBlock & iLBlock,unsigned int runnumber,unsigned int luminum){
  //std::cout<<"writing runnumber,luminum "<<runnumber<<" "<<luminum<<std::endl;
  auto pIn1 = std::make_unique<LumiSummary>();
  auto pIn2 = std::make_unique<LumiDetails>();
  if(m_isNullRun){
    pIn1->setLumiVersion("-1");
    pIn2->setLumiVersion("-1");
    iLBlock.put(std::move(pIn1));
    iLBlock.put(std::move(pIn2));
    return;
  }
  PerLSData& lsdata=m_lscache[luminum];
  pIn1->setLumiData(lsdata.lumivalue,lsdata.lumierror,lsdata.lumiquality);
  pIn1->setDeadCount(lsdata.deadcount);
  if(!lsdata.l1data.empty()){
    //std::cout<<"bitzerocount "<<lsdata.bitzerocount<<std::endl;
    //std::cout<<"bitzeroprescale "<<lsdata.bitzeroprescale<<std::endl;
    //std::cout<<"product "<<lsdata.bitzerocount*lsdata.bitzeroprescale<<std::endl;
    pIn1->setBitZeroCount(lsdata.bitzerocount*lsdata.bitzeroprescale);
  }
  pIn1->setlsnumber(luminum);
  pIn1->setOrbitData(lsdata.startorbit,lsdata.numorbit);
  std::vector<LumiSummary::L1> l1temp;
  for(std::vector< L1Data >::iterator it=lsdata.l1data.begin();it!=lsdata.l1data.end();++it){
    LumiSummary::L1 trgtmp;
    trgtmp.triggernameidx=m_runcache.TRGBitNameToIndex[it->bitname];
    trgtmp.prescale=it->prescale;
    l1temp.push_back(trgtmp);
  }
  std::vector<LumiSummary::HLT> hlttemp;
  for(std::vector< HLTData >::iterator it=lsdata.hltdata.begin();it!=lsdata.hltdata.end();++it){
    LumiSummary::HLT hlttmp;
    hlttmp.pathnameidx=m_runcache.HLTPathNameToIndex[it->pathname];;
    hlttmp.prescale=it->prescale;
    hlttemp.push_back(hlttmp);
  }
  pIn1->swapL1Data(l1temp);
  pIn1->swapHLTData(hlttemp);
  pIn1->setLumiVersion(m_lumiversion);  
  pIn2->fillBeamIntensities(lsdata.beam1intensity,lsdata.beam2intensity);
  for(unsigned int i=0;i<lsdata.bunchlumivalue.size();++i){
    std::string algoname=lsdata.bunchlumivalue[i].first;
    if(algoname=="OCC1"){
      pIn2->fill(LumiDetails::kOCC1,lsdata.bunchlumivalue[i].second,lsdata.bunchlumierror[i].second,lsdata.bunchlumiquality[i].second);
    }else if(algoname=="OCC2"){      
      pIn2->fill(LumiDetails::kOCC2,lsdata.bunchlumivalue[i].second,lsdata.bunchlumierror[i].second,lsdata.bunchlumiquality[i].second);
    }else if(algoname=="ET"){
      pIn2->fill(LumiDetails::kET,lsdata.bunchlumivalue[i].second,lsdata.bunchlumierror[i].second,lsdata.bunchlumiquality[i].second);
    }else if(algoname=="PLT"){
      pIn2->fill(LumiDetails::kPLT,lsdata.bunchlumivalue[i].second,lsdata.bunchlumierror[i].second,lsdata.bunchlumiquality[i].second);
    }
  }
  pIn2->setLumiVersion(m_lumiversion);
  iLBlock.put(std::move(pIn1));
  iLBlock.put(std::move(pIn2));
}
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(LumiProducer);

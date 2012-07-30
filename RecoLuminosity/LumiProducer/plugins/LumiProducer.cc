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
// $Id: LumiProducer.cc,v 1.18 2011/01/17 11:01:50 xiezhen Exp $

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
    std::vector< std::pair<std::string, std::vector<float> > > bunchlumivalue;
    std::vector< std::pair<std::string, std::vector<float> > > bunchlumierror;
    std::vector< std::pair<std::string, std::vector<short> > > bunchlumiquality;
    std::vector<float> beam1intensity;
    std::vector<float> beam2intensity;
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
  void fillLSCache(unsigned int luminum);
  void writeProductsForEntry(edm::LuminosityBlock & iLBlock,unsigned int runnumber,unsigned int luminum);
  const std::string servletTranslation(const std::string& servlet) const;
  std::string x2s(const XMLCh* input)const;
  XMLCh* s2x(const std::string& input)const;
  std::string toParentString(const xercesc::DOMNode &nodeToConvert)const;

  std::string m_connectStr;
  std::string m_lumiversion;
  std::string m_siteconfpath;
  unsigned int m_cachedrun;
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
LumiProducer::LumiProducer(const edm::ParameterSet& iConfig):m_cachedrun(0),m_isNullRun(false),m_cachesize(0)
{
  // register your products
  produces<LumiSummary, edm::InLumi>();
  produces<LumiDetails, edm::InLumi>();
  // set up cache
  std::string connectStr=iConfig.getParameter<std::string>("connect");
  m_cachesize=iConfig.getUntrackedParameter<unsigned int>("ncacheEntries",5);
  m_lumiversion=iConfig.getUntrackedParameter<std::string>("lumiversion");
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
  m_cachedrun=runnumber;
  fillRunCache(runnumber);
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
  if(m_lscache.find(luminum)==m_lscache.end()){
    //if runnumber is cached but LS is not, this is the first LS, fill LS cache to full capacity
    fillLSCache(luminum);
  }
  //here the presence of ls is guaranteed
  writeProductsForEntry(iLBlock,runnumber,luminum); 
}
void 
LumiProducer::fillRunCache(unsigned int runnumber){
  //queries once per run
  edm::Service<lumi::service::DBService> mydbservice;
  if( !mydbservice.isAvailable() ){
    throw cms::Exception("Non existing service lumi::service::DBService");
  }
  //std::cout<<"in fillRunCache "<<runnumber<<std::endl;
  coral::ISessionProxy* session=mydbservice->connectReadOnly(m_connectStr);
  try{
    session->transaction().start(true);
    coral::ISchema& schema=session->nominalSchema();
    //
    //select bitnum,bitname from trg where runnum=:runnum and cmslsnum=:1 order by bitnum;
    //
    //std::cout<<"got schema handle "<<std::endl;
    m_cachedrun=runnumber;
    coral::AttributeList trgBindVariables;
    trgBindVariables.extend("runnum",typeid(unsigned int));
    trgBindVariables.extend("cmslsnum",typeid(unsigned int));
    trgBindVariables["runnum"].data<unsigned int>()=runnumber;
    trgBindVariables["cmslsnum"].data<unsigned int>()=1;
    coral::AttributeList trgOutput;
    trgOutput.extend("bitnum",typeid(unsigned int));
    trgOutput.extend("bitname",typeid(std::string));
    coral::IQuery* trgQuery=schema.newQuery();
    trgQuery->addToTableList(lumi::LumiNames::trgTableName());
    trgQuery->addToOutputList("BITNUM");
    trgQuery->addToOutputList("BITNAME");
    trgQuery->setCondition("RUNNUM=:runnum AND CMSLSNUM=:cmslsnum",trgBindVariables);
    trgQuery->defineOutput(trgOutput);
    coral::ICursor& trgcursor=trgQuery->execute();
    unsigned int rowcounter=0;
    while( trgcursor.next() ){
      const coral::AttributeList& row=trgcursor.currentRow();
      m_runcache.TRGBitNames.insert(std::make_pair(row["bitnum"].data<unsigned int>(),row["bitname"].data<std::string>()));
      ++rowcounter;
    }
    delete trgQuery;
    if (rowcounter==0){
      m_isNullRun=true;
      session->transaction().commit();
      mydbservice->disconnect(session);
      return;
    }
    //
    //select pathname from from hlt where  runnum=:runnum and cmslsnum=:1 order by pathname;
    //
    coral::AttributeList hltBindVariables;
    hltBindVariables.extend("runnum",typeid(unsigned int));
    hltBindVariables.extend("cmslsnum",typeid(unsigned int));
    hltBindVariables["runnum"].data<unsigned int>()=runnumber;
    hltBindVariables["cmslsnum"].data<unsigned int>()=1;
    coral::AttributeList hltOutput;
    hltOutput.extend("pathname",typeid(std::string));
    coral::IQuery* hltQuery=schema.newQuery();
    hltQuery->addToTableList(lumi::LumiNames::hltTableName());
    hltQuery->addToOutputList("PATHNAME");
    hltQuery->setCondition("RUNNUM=:runnum AND CMSLSNUM=:cmslsnum",hltBindVariables);
    hltQuery->addToOrderList("PATHNAME");
    hltQuery->defineOutput(hltOutput);
    coral::ICursor& hltcursor=hltQuery->execute();
    rowcounter=0;
    unsigned int pathcount=0;
    while( hltcursor.next() ){
      const coral::AttributeList& row=hltcursor.currentRow();
      m_runcache.HLTPathNames.insert(std::make_pair(pathcount,row["pathname"].data<std::string>()));
      ++pathcount;
      ++rowcounter;
    }
    delete hltQuery;   
    if (rowcounter==0){
      m_isNullRun=true;
      session->transaction().commit();
      mydbservice->disconnect(session);
      return;
    }
    session->transaction().commit();
  }catch(const coral::Exception& er){
    session->transaction().rollback();
    mydbservice->disconnect(session);
    throw cms::Exception("DatabaseError ")<<er.what();
  }
  mydbservice->disconnect(session);
}
void
LumiProducer::fillLSCache(unsigned int luminum){
  //std::cout<<"in fillLSCache "<<luminum<<std::endl;
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
  //select cmslsnum,instlumi,instlumierror,lumiquality,startorbit,numorbit,bxindex,beam1intensity,beam2intensity from lumisummary where cmslsnum>=:lsmin and cmslsnum<:lsmax and runnum=:runnumber ;
  //
  edm::Service<lumi::service::DBService> mydbservice;
  if( !mydbservice.isAvailable() ){
    throw cms::Exception("Non existing service lumi::service::DBService");
  }
  coral::ISessionProxy* session=mydbservice->connectReadOnly(m_connectStr);
  try{
    session->transaction().start(true);
    coral::ISchema& schema=session->nominalSchema();
    coral::AttributeList lumisummaryBindVariables;
    lumisummaryBindVariables.extend("runnum",typeid(unsigned int));
    lumisummaryBindVariables.extend("lsmin",typeid(unsigned int));
    lumisummaryBindVariables.extend("lsmax",typeid(unsigned int));
  
    lumisummaryBindVariables["runnum"].data<unsigned int>()=m_cachedrun;
    lumisummaryBindVariables["lsmin"].data<unsigned int>()=luminum;
    lumisummaryBindVariables["lsmax"].data<unsigned int>()=luminum+m_cachesize;
    coral::AttributeList lumisummaryOutput;
    lumisummaryOutput.extend("cmslsnum",typeid(unsigned int));
    lumisummaryOutput.extend("instlumi",typeid(float));
    lumisummaryOutput.extend("instlumierror",typeid(float));
    lumisummaryOutput.extend("instlumiquality",typeid(short));
    lumisummaryOutput.extend("startorbit",typeid(unsigned int));
    lumisummaryOutput.extend("numorbit",typeid(unsigned int));
    lumisummaryOutput.extend("bxindexBlob",typeid(coral::Blob));
    lumisummaryOutput.extend("beam1intensityBlob",typeid(coral::Blob));
    lumisummaryOutput.extend("beam2intensityBlob",typeid(coral::Blob));
    
    coral::IQuery* lumisummaryQuery=schema.newQuery();
    lumisummaryQuery->addToTableList(lumi::LumiNames::lumisummaryTableName());
    lumisummaryQuery->addToOutputList("CMSLSNUM","cmslsnum");
    lumisummaryQuery->addToOutputList("INSTLUMI","instlumi");
    lumisummaryQuery->addToOutputList("INSTLUMIERROR","instlumierror");
    lumisummaryQuery->addToOutputList("INSTLUMIQUALITY","instlumiquality");
    lumisummaryQuery->addToOutputList("STARTORBIT","startorbit");
    lumisummaryQuery->addToOutputList("NUMORBIT","numorbit");
    lumisummaryQuery->addToOutputList("CMSBXINDEXBLOB","bxindexBlob");
    lumisummaryQuery->addToOutputList("BEAMINTENSITYBLOB_1","beam1intensityBlob");
    lumisummaryQuery->addToOutputList("BEAMINTENSITYBLOB_2","beam2intensityBlob");
    lumisummaryQuery->setCondition("RUNNUM=:runnum AND CMSLSNUM>=:lsmin AND CMSLSNUM<:lsmax",lumisummaryBindVariables);
    lumisummaryQuery->defineOutput(lumisummaryOutput);
    coral::ICursor& lumisummarycursor=lumisummaryQuery->execute();
    unsigned int rowcounter=0;
    while( lumisummarycursor.next() ){
      const coral::AttributeList& row=lumisummarycursor.currentRow();
      unsigned int cmslsnum=row["cmslsnum"].data<unsigned int>();
      //std::cout<<"cmslsnum "<<cmslsnum<<std::endl;
      PerLSData& lsdata=m_lscache[cmslsnum];
      lsdata.lumivalue=row["instlumi"].data<float>();
      lsdata.lumierror=row["instlumierror"].data<float>();
      lsdata.lumiquality=row["instlumiquality"].data<short>();
      lsdata.startorbit=row["startorbit"].data<unsigned int>();
      lsdata.numorbit=row["numorbit"].data<unsigned int>();
      
      if(!row["bxindexBlob"].isNull()){
	const coral::Blob& bxindexBlob=row["bxindexBlob"].data<coral::Blob>();
	const void* bxindex_StartAddress=bxindexBlob.startingAddress();
	short* bxindex=(short*)::malloc(bxindexBlob.size());
	const coral::Blob& beam1intensityBlob=row["beam1intensityBlob"].data<coral::Blob>();
	const void* beam1intensityBlob_StartAddress=beam1intensityBlob.startingAddress();
	float* beam1intensity=(float*)::malloc(beam1intensityBlob.size());
	const coral::Blob& beam2intensityBlob=row["beam2intensityBlob"].data<coral::Blob>();
	const void* beam2intensityBlob_StartAddress=beam2intensityBlob.startingAddress();
	float* beam2intensity=(float*)::malloc(beam2intensityBlob.size());
	std::memmove(bxindex,bxindex_StartAddress,bxindexBlob.size());
	std::memmove(beam1intensity,beam1intensityBlob_StartAddress,beam1intensityBlob.size());
	std::memmove(beam2intensity,beam2intensityBlob_StartAddress,beam2intensityBlob.size());
	//std::cout<<"lsnum,pos,bxidx,beam1intensity,beam2intensity "<<std::endl;
	for(unsigned int i=0;i<bxindexBlob.size()/sizeof(short);++i){
	  unsigned int idx=bxindex[i];
	  lsdata.beam1intensity.at(idx)=beam1intensity[i];
	  lsdata.beam2intensity.at(idx)=beam2intensity[i];
	  //std::cout<<cmslsnum<<","<<i<<","<<idx<<","<<beam1intensity[i]<<","<<beam2intensity[i]<<std::endl;
	}
	::free(bxindex);
	::free(beam1intensity);
	::free(beam2intensity);
      }
      ++rowcounter;
    }
    if (rowcounter==0){
      m_isNullRun=true;
      return;
    }
    delete lumisummaryQuery;
    
    //
    //select lumisummary.cmslsnum,lumidetail.bxlumivalue,lumidetail.bxlumierror,lumidetail.bxlumiquality,lumidetail.algoname from lumisummary,lumidetail where lumisummary.lumisummary_id=lumidetail.lumisummary_id and lumisummary.runnum=:runnum and lumisummary.cmslsnum>=:luminum and lumisummary.cmslsnum<:luminum+cachesize order by lumidetail.algoname,lumisummary.cmslsnum
    //
    coral::AttributeList lumidetailBindVariables;
    lumidetailBindVariables.extend("runnum",typeid(unsigned int));
    lumidetailBindVariables.extend("lsmin",typeid(unsigned int));
    lumidetailBindVariables.extend("lsmax",typeid(unsigned int));
    
    lumidetailBindVariables["runnum"].data<unsigned int>()=m_cachedrun;
    lumidetailBindVariables["lsmin"].data<unsigned int>()=luminum;
    lumidetailBindVariables["lsmax"].data<unsigned int>()=luminum+m_cachesize;
    coral::AttributeList lumidetailOutput;
    
    lumidetailOutput.extend("cmslsnum",typeid(unsigned int));
    lumidetailOutput.extend("bxlumivalue",typeid(coral::Blob));
    lumidetailOutput.extend("bxlumierror",typeid(coral::Blob));
    lumidetailOutput.extend("bxlumiquality",typeid(coral::Blob));
    lumidetailOutput.extend("algoname",typeid(std::string));

    coral::IQuery* lumidetailQuery=schema.newQuery();
    lumidetailQuery->addToTableList(lumi::LumiNames::lumisummaryTableName());
    lumidetailQuery->addToTableList(lumi::LumiNames::lumidetailTableName());
    lumidetailQuery->addToOutputList(lumi::LumiNames::lumisummaryTableName()+".CMSLSNUM","cmslsnum");
    lumidetailQuery->addToOutputList(lumi::LumiNames::lumidetailTableName()+".BXLUMIVALUE","bxlumivalue");
    lumidetailQuery->addToOutputList(lumi::LumiNames::lumidetailTableName()+".BXLUMIERROR","bxlumierror");
    lumidetailQuery->addToOutputList(lumi::LumiNames::lumidetailTableName()+".BXLUMIQUALITY","instlumiquality");
    lumidetailQuery->addToOutputList(lumi::LumiNames::lumidetailTableName()+".ALGONAME","algoname");
    lumidetailQuery->setCondition(lumi::LumiNames::lumisummaryTableName()+".LUMISUMMARY_ID="+lumi::LumiNames::lumidetailTableName()+".LUMISUMMARY_ID AND "+lumi::LumiNames::lumisummaryTableName()+".RUNNUM=:runnum AND "+lumi::LumiNames::lumisummaryTableName()+".CMSLSNUM>=:lsmin AND "+lumi::LumiNames::lumisummaryTableName()+".CMSLSNUM<:lsmax",lumidetailBindVariables);
    lumidetailQuery->addToOrderList(lumi::LumiNames::lumidetailTableName()+".ALGONAME");
    lumidetailQuery->addToOrderList(lumi::LumiNames::lumisummaryTableName()+".CMSLSNUM");
    lumidetailQuery->defineOutput(lumidetailOutput);
    coral::ICursor& lumidetailcursor=lumidetailQuery->execute();
    while( lumidetailcursor.next() ){
      const coral::AttributeList& row=lumidetailcursor.currentRow();
      unsigned int cmslsnum=row["cmslsnum"].data<unsigned int>();
      std::string algoname=row["algoname"].data<std::string>();
      //std::cout<<"cmslsnum "<<cmslsnum<<" "<<algoname<<std::endl;
      PerLSData& lsdata=m_lscache[cmslsnum];
      if( !row["bxlumivalue"].isNull() && !row["bxlumierror"].isNull() && !row["bxlumiquality"].isNull() ){
	const coral::Blob& bxlumivalueBlob=row["bxlumivalue"].data<coral::Blob>();
	const coral::Blob& bxlumierrorBlob=row["bxlumierror"].data<coral::Blob>();
	const coral::Blob& bxlumiqualityBlob=row["bxlumiquality"].data<coral::Blob>();
	const void* bxlumivalueBlob_StartAddress=bxlumivalueBlob.startingAddress();
	const void* bxlumierrorBlob_StartAddress=bxlumierrorBlob.startingAddress();
	const void* bxlumiqualityBlob_StartAddress=bxlumiqualityBlob.startingAddress();
	float* bxlumivalue=(float*)::malloc(bxlumivalueBlob.size());
	float* bxlumierror=(float*)::malloc(bxlumierrorBlob.size());
	short* bxlumiquality=(short*)::malloc(bxlumiqualityBlob.size());
	std::memmove(bxlumivalue,bxlumivalueBlob_StartAddress,bxlumivalueBlob.size());
	std::memmove(bxlumierror,bxlumierrorBlob_StartAddress,bxlumierrorBlob.size());
	std::memmove(bxlumiquality,bxlumiqualityBlob_StartAddress,bxlumiqualityBlob.size());
	std::vector<float> bxlumivalueVec(bxlumivalue,bxlumivalue+bxlumivalueBlob.size()/sizeof(float));
	::free(bxlumivalue);
	lsdata.bunchlumivalue.push_back(std::make_pair(algoname,bxlumivalueVec));
	std::vector<float> bxlumierrorVec(bxlumierror,bxlumierror+bxlumierrorBlob.size()/sizeof(float));
	::free(bxlumierror);
	lsdata.bunchlumierror.push_back(std::make_pair(algoname,bxlumierrorVec));
	std::vector<short> bxlumiqualityVec(bxlumiquality,bxlumiquality+bxlumiqualityBlob.size()/sizeof(short));
	lsdata.bunchlumiquality.push_back(std::make_pair(algoname,bxlumiqualityVec));
	::free(bxlumiquality);
      }
    }
    delete lumidetailQuery;
    //
    //select cmslsnum,bitnum,deadtime,prescale,trgcount from trg where cmslsnum >=:luminum and cmslsnum<:luminum+cachesize AND runnum=:runnum order by cmslsnum,bitnum
    //
    coral::AttributeList trgBindVariables;
    trgBindVariables.extend("runnum",typeid(unsigned int));
    trgBindVariables.extend("lsmin",typeid(unsigned int));
    trgBindVariables.extend("lsmax",typeid(unsigned int));
    trgBindVariables["runnum"].data<unsigned int>()=m_cachedrun;
    trgBindVariables["lsmin"].data<unsigned int>()=luminum;
    trgBindVariables["lsmax"].data<unsigned int>()=luminum+m_cachesize;
    coral::AttributeList trgOutput;
    trgOutput.extend("cmslsnum",typeid(unsigned int));
    trgOutput.extend("bitnum",typeid(unsigned int));
    trgOutput.extend("deadtime",typeid(unsigned long long));
    trgOutput.extend("prescale",typeid(unsigned int));
    trgOutput.extend("trgcount",typeid(unsigned int));
    
    coral::IQuery* trgQuery=schema.newQuery();
    trgQuery->addToTableList(lumi::LumiNames::trgTableName());
    trgQuery->addToOutputList("CMSLSNUM","cmslsnum");
    trgQuery->addToOutputList("BITNUM","bitnum");
    trgQuery->addToOutputList("DEADTIME","deadtime");
    trgQuery->addToOutputList("PRESCALE","prescale");
    trgQuery->addToOutputList("TRGCOUNT","trgcount");
    trgQuery->setCondition("CMSLSNUM>=:lsmin AND CMSLSNUM<:lsmax AND RUNNUM=:runnum ",trgBindVariables);
    trgQuery->addToOrderList("CMSLSNUM");
    trgQuery->addToOrderList("BITNUM");
    trgQuery->defineOutput(trgOutput);
    coral::ICursor& trgcursor=trgQuery->execute();
    while( trgcursor.next() ){
      const coral::AttributeList& row=trgcursor.currentRow();
      unsigned int cmslsnum=row["cmslsnum"].data<unsigned int>();
      PerLSData& lsdata=m_lscache[cmslsnum];
      lsdata.deadcount=row["deadtime"].data<unsigned long long>();
      L1Data l1tmp;
      l1tmp.bitnum=row["bitnum"].data<unsigned int>();
      l1tmp.prescale=row["prescale"].data<unsigned int>();
      l1tmp.ratecount=row["trgcount"].data<unsigned int>();
      lsdata.l1data.push_back(l1tmp);
    }
    delete trgQuery;
    //
    //select cmslsnum,inputcount,acceptcount,prescale from hlt where cmslsnum >=:luminum and cmslsnum<=:luminum+cachesize and runnum=:runnumber order by cmslsum,pathname
    //
    coral::AttributeList hltBindVariables;
    hltBindVariables.extend("runnum",typeid(unsigned int));
    hltBindVariables.extend("lsmin",typeid(unsigned int));
    hltBindVariables.extend("lsmax",typeid(unsigned int));
    hltBindVariables["runnum"].data<unsigned int>()=m_cachedrun;
    hltBindVariables["lsmin"].data<unsigned int>()=luminum;
    hltBindVariables["lsmax"].data<unsigned int>()=luminum+m_cachesize;
    coral::AttributeList hltOutput;
    hltOutput.extend("cmslsnum",typeid(unsigned int));
    hltOutput.extend("inputcount",typeid(unsigned int));
    hltOutput.extend("acceptcount",typeid(unsigned int));
    hltOutput.extend("prescale",typeid(unsigned int));
    coral::IQuery* hltQuery=schema.newQuery();
    hltQuery->addToTableList(lumi::LumiNames::hltTableName());
    hltQuery->addToOutputList("CMSLSNUM","cmslsnum");
    hltQuery->addToOutputList("INPUTCOUNT","inputcount");
    hltQuery->addToOutputList("ACCEPTCOUNT","acceptcount");
    hltQuery->addToOutputList("PRESCALE","prescale");
    hltQuery->setCondition("CMSLSNUM>=:lsmin AND CMSLSNUM<:lsmax AND RUNNUM=:runnum",hltBindVariables);
    hltQuery->addToOrderList("CMSLSNUM");
    hltQuery->addToOrderList("PATHNAME");
    hltQuery->defineOutput(hltOutput);
    coral::ICursor& hltcursor=hltQuery->execute();
    unsigned int npaths=m_runcache.HLTPathNames.size();
    unsigned int pathcount=0;
    while( hltcursor.next() ){
      const coral::AttributeList& row=hltcursor.currentRow();   
      unsigned int cmslsnum=row["cmslsnum"].data<unsigned int>();
      PerLSData& lsdata=m_lscache[cmslsnum];
      HLTData hlttmp;
      hlttmp.pathnum=pathcount;
      hlttmp.prescale=row["prescale"].data<unsigned int>();
      hlttmp.l1passcount=row["inputcount"].data<unsigned int>();
      hlttmp.acceptcount=row["acceptcount"].data<unsigned int>();
      lsdata.hltdata.push_back(hlttmp);
      if(pathcount!=npaths){
	++pathcount;
      }else{
	pathcount=0;
      }
    }
    delete hltQuery;
    session->transaction().commit();
  }catch(const coral::Exception& er){
    session->transaction().rollback();
    mydbservice->disconnect(session);
    throw cms::Exception("DatabaseError ")<<er.what();
  }
  mydbservice->disconnect(session);
}
void
LumiProducer::writeProductsForEntry(edm::LuminosityBlock & iLBlock,unsigned int runnumber,unsigned int luminum){
  std::auto_ptr<LumiSummary> pOut1;
  std::auto_ptr<LumiDetails> pOut2;
  LumiSummary* pIn1=new LumiSummary;
  LumiDetails* pIn2=new LumiDetails;
  if(m_isNullRun){
    pIn1->setLumiVersion("-1");
    pIn2->setLumiVersion("-1");
    pOut1.reset(pIn1);
    iLBlock.put(pOut1);
    pOut2.reset(pIn2);
    iLBlock.put(pOut2);
    return;
  }
  PerLSData& lsdata=m_lscache[luminum];
  pIn1->setLumiData(lsdata.lumivalue,lsdata.lumierror,lsdata.lumiquality);
  pIn1->setDeadtime(lsdata.deadcount);
  pIn1->setlsnumber(luminum);
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
  pOut1.reset(pIn1);
  iLBlock.put(pOut1);
  pOut2.reset(pIn2);
  iLBlock.put(pOut2);
}
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(LumiProducer);

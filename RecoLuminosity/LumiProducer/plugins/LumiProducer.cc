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

  //edm::ParameterSet pset_;

  std::string m_connectStr;
  std::string m_lumiversion;
  std::string m_siteconfpath;
  const std::string servletTranslation(const std::string& servlet) const;
  std::string x2s(const XMLCh* input)const;
  XMLCh* s2x(const std::string& input)const;
  std::string toParentString(const xercesc::DOMNode &nodeToConvert)const;

private:
  bool m_isNullRun;
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
LumiProducer::LumiProducer(const edm::ParameterSet& iConfig)
{
  // register your products
  produces<LumiSummary, edm::InLumi>();
  produces<LumiDetails, edm::InLumi>();
  std::string connectStr=iConfig.getParameter<std::string>("connect");
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
  m_lumiversion=iConfig.getUntrackedParameter<std::string>("lumiversion");
  m_isNullRun=false;
}

LumiProducer::~LumiProducer(){ 
}
//
// member functions
//
void LumiProducer::produce(edm::Event& e, const edm::EventSetup& iSetup){ 
}
/**
void LumiProducer::fillDefaultLumi(edm::LuminosityBlock &iLBlock){
  LumiSummary* pIn1=new LumiSummary;
  std::auto_ptr<LumiSummary> pOut1(pIn1);
  iLBlock.put(pOut1);
  LumiDetails* pIn2=new LumiDetails;
  std::auto_ptr<LumiDetails> pOut2(pIn2);
  iLBlock.put(pOut2);
}
**/
void LumiProducer::beginRun(edm::Run& run,edm::EventSetup const &iSetup){
  /**
   at the beginning of run, we check
   if all required tables exist
   if lumi,trg,hlt data all exist for the run
   if not set invalid flag and no more db access for this run
   here 7 queries per beginRun method
  **/
  edm::Service<lumi::service::DBService> mydbservice;
  if( !mydbservice.isAvailable() ){
    throw cms::Exception("Non existing service lumi::service::DBService");
  }
  unsigned int runnumber=run.run();
  coral::ISessionProxy* session=mydbservice->connectReadOnly(m_connectStr);
  try{
    session->transaction().start(true);
    coral::ISchema& schema=session->nominalSchema();
    if(!schema.existsTable(lumi::LumiNames::lumisummaryTableName())){
      throw lumi::Exception(std::string("non-existing table ")+lumi::LumiNames::lumisummaryTableName(),"beginRun","LumiProducer");
    }
    if(!schema.existsTable(lumi::LumiNames::lumidetailTableName())){
      throw lumi::Exception(std::string("non-existing table ")+lumi::LumiNames::lumisummaryTableName(),"beginRun","LumiProducer");
    }
    if(!schema.existsTable(lumi::LumiNames::trgTableName())){
      throw lumi::Exception(std::string("non-existing table ")+lumi::LumiNames::trgTableName(),"beginRun","LumiProducer");
    }
    if(!schema.existsTable(lumi::LumiNames::hltTableName())){
      throw lumi::Exception(std::string("non-existing table ")+lumi::LumiNames::hltTableName(),"beginRun","LumiProducer");
    }
    //
    //select count(*) from lumisummary where runnum=:runnumber
    //select count(*) from trg where runnum=:runnumber 
    //select count(*) from hlt where runnum=:runnumber
    //
    coral::AttributeList bindVariables;
    bindVariables.extend("runnumber",typeid(unsigned int));
    bindVariables["runnumber"].data<unsigned int>()=runnumber;

    coral::AttributeList lumiResult;
    lumiResult.extend("lumisize",typeid(unsigned int));
    coral::IQuery* lumiQuery=schema.tableHandle(lumi::LumiNames::lumisummaryTableName()).newQuery();
    lumiQuery->addToOutputList("count(RUNNUM)","lumisize");
    lumiQuery->setCondition("RUNNUM=:runnumber",bindVariables);
    lumiQuery->defineOutput(lumiResult);
    coral::ICursor& lumicursor=lumiQuery->execute();
    unsigned int nlumirun=0;
    while( lumicursor.next() ){
      const coral::AttributeList& row=lumicursor.currentRow();
      nlumirun=row["lumisize"].data<unsigned int>();
    }
    delete lumiQuery;
    if (nlumirun==0){
      m_isNullRun=true;
      session->transaction().commit();
      mydbservice->disconnect(session);
      return;
    }
    coral::AttributeList trgResult;
    trgResult.extend("trgsize",typeid(unsigned int));
    coral::IQuery* trgQuery=schema.tableHandle(lumi::LumiNames::trgTableName()).newQuery();
    trgQuery->addToOutputList("count(*)","trgsize");
    trgQuery->setCondition("RUNNUM=:runnumber",bindVariables);
    trgQuery->defineOutput(trgResult);
    coral::ICursor& trgcursor=trgQuery->execute();
    unsigned int ntrgrun=0; 
    while( trgcursor.next() ){
      const coral::AttributeList& row=trgcursor.currentRow();
      ntrgrun=row["trgsize"].data<unsigned int>();
    }
    delete trgQuery;
    if (ntrgrun==0){
      m_isNullRun=true;
      session->transaction().commit();
      mydbservice->disconnect(session);
      return;
    }
    
    coral::AttributeList hltResult;
    hltResult.extend("hltsize",typeid(unsigned int));
    coral::IQuery* hltQuery=schema.tableHandle(lumi::LumiNames::hltTableName()).newQuery();
    hltQuery->addToOutputList("count(*)","hltsize");
    hltQuery->setCondition("RUNNUM=:runnumber",bindVariables);
    hltQuery->defineOutput(hltResult);
    coral::ICursor& hltcursor=hltQuery->execute();
    unsigned int nhltrun=0; 
    while( hltcursor.next() ){
      const coral::AttributeList& row=hltcursor.currentRow();
      nhltrun=row["hltsize"].data<unsigned int>();
    }
    delete hltQuery;
    if (nhltrun==0){
      m_isNullRun=true;
      session->transaction().commit();
      mydbservice->disconnect(session);
      return;
    }
  }catch( const coral::Exception& er){
    session->transaction().rollback();
    mydbservice->disconnect(session);
    throw er;
  }
  session->transaction().commit();
  mydbservice->disconnect(session);
}
void LumiProducer::beginLuminosityBlock(edm::LuminosityBlock &iLBlock, edm::EventSetup const &iSetup) {  
}
void LumiProducer::endLuminosityBlock(edm::LuminosityBlock & iLBlock, 
				      edm::EventSetup const& c){
  /**
     here 4 queries per endLuminosityBlock  method
   **/
  std::auto_ptr<LumiSummary> pOut1;
  std::auto_ptr<LumiDetails> pOut2;
  LumiSummary* pIn1=new LumiSummary;
  LumiDetails* pIn2=new LumiDetails;
  if(m_isNullRun){
    pOut1.reset(pIn1);
    iLBlock.put(pOut1);
    pOut2.reset(pIn2);
    iLBlock.put(pOut2);
    return;
  }
  unsigned int runnumber=iLBlock.run();
  unsigned int luminum=iLBlock.luminosityBlock();
  edm::Service<lumi::service::DBService> mydbservice;
  if( !mydbservice.isAvailable() ){
    //std::cout<<"Service is unavailable"<<std::endl;
    return;
  }
  coral::ISessionProxy* session=mydbservice->connectReadOnly(m_connectStr);
  coral::ITypeConverter& tpc=session->typeConverter();
  tpc.setCppTypeForSqlType("short","NUMBER(7)");
  tpc.setCppTypeForSqlType("unsigned int","NUMBER(10)");
  tpc.setCppTypeForSqlType("unsigned long long","NUMBER(20)");
  try{
    session->transaction().start(true);
    coral::ISchema& schema=session->nominalSchema();
    //
    //select cmslsnum,lumisummary_id,instlumi,instlumierror,lumisectionquality,startorbit,numorbit from LUMISUMMARY where runnum=:runnumber AND cmslsnum=:cmslsnum AND cmsalive=:cmsalive AND lumiversion=:lumiversion order by cmslsnum
    //
    coral::AttributeList lumiBindVariables;
    lumiBindVariables.extend("runnumber",typeid(unsigned int));
    lumiBindVariables.extend("cmslsnum",typeid(unsigned int));
    lumiBindVariables.extend("lumiversion",typeid(std::string));
    lumiBindVariables.extend("cmsalive",typeid(short));

    lumiBindVariables["runnumber"].data<unsigned int>()=runnumber;
    lumiBindVariables["cmslsnum"].data<unsigned int>()=luminum;
    lumiBindVariables["lumiversion"].data<std::string>()=m_lumiversion;
    lumiBindVariables["cmsalive"].data<short>()=1;

    coral::AttributeList lumiOutput;
    lumiOutput.extend("cmslsnum",typeid(unsigned int));
    lumiOutput.extend("lumisummary_id",typeid(unsigned long long));
    lumiOutput.extend("instlumi",typeid(float));
    lumiOutput.extend("instlumierror",typeid(float));
    lumiOutput.extend("lumisectionquality",typeid(short));
    lumiOutput.extend("startorbit",typeid(unsigned int));
    lumiOutput.extend("numorbit",typeid(unsigned int));
    
    coral::IQuery* lumiQuery=schema.tableHandle(lumi::LumiNames::lumisummaryTableName()).newQuery();
    lumiQuery->addToOutputList("CMSLSNUM");
    lumiQuery->addToOutputList("LUMISUMMARY_ID");
    lumiQuery->addToOutputList("INSTLUMI");
    lumiQuery->addToOutputList("INSTLUMIERROR");
    lumiQuery->addToOutputList("LUMISECTIONQUALITY");
    lumiQuery->addToOutputList("STARTORBIT");
    lumiQuery->addToOutputList("NUMORBIT");
    lumiQuery->addToOrderList("CMSLSNUM");
    lumiQuery->setCondition("RUNNUM =:runnumber AND CMSLSNUM =:cmslsnum AND CMSALIVE=:cmsalive AND LUMIVERSION=:lumiversion",lumiBindVariables);
    lumiQuery->defineOutput(lumiOutput);
    coral::ICursor& lumicursor=lumiQuery->execute();
    unsigned long long lumisummary_id=0;
    float instlumi=0.0;
    float instlumierror=0.0;
    short lumisectionquality=0;
    unsigned int startorbit=0;
    unsigned int numorbit=0;
    unsigned int s=0;
    while( lumicursor.next() ){
      const coral::AttributeList& row=lumicursor.currentRow();     
      //row.toOutputStream( std::cout ) << std::endl;
      lumisummary_id=row["lumisummary_id"].data<unsigned long long>();
      instlumi=row["instlumi"].data<float>();
      instlumierror=row["instlumierror"].data<float>();
      lumisectionquality=row["lumisectionquality"].data<short>();
      startorbit=row["startorbit"].data<unsigned int>();
      numorbit=row["numorbit"].data<unsigned int>();
      ++s;
    }
    delete lumiQuery;
    pIn1->setLumiVersion(m_lumiversion);
    pIn1->setlsnumber(luminum);
    pIn1->setLumiData(instlumi,instlumierror,lumisectionquality);
    pIn1->setOrbitData(startorbit,numorbit);
    if(s==0){//if no result, meaning LS missing, fill default value everywhere and return right away
      pOut1.reset(pIn1);
      iLBlock.put(pOut1);
      pOut2.reset(pIn2);
      iLBlock.put(pOut2);
      session->transaction().commit();
      mydbservice->disconnect(session);
      return;
    }
    //
    //select bxlumivalue,bxlumierror,bxlumiquality,algoname from LUMIDETAIL where lumisummary_id=:lumisummary_id 
    //
    coral::AttributeList detailBindVariables;
    detailBindVariables.extend("lumisummary_id",typeid(unsigned long long));
    detailBindVariables["lumisummary_id"].data<unsigned long long>()=lumisummary_id;
    coral::AttributeList detailOutput;
    detailOutput.extend("bxlumivalue",typeid(coral::Blob));
    detailOutput.extend("bxlumierror",typeid(coral::Blob));
    detailOutput.extend("bxlumiquality",typeid(coral::Blob));
    detailOutput.extend("algoname",typeid(std::string));
    
    coral::IQuery* detailQuery=schema.tableHandle(lumi::LumiNames::lumidetailTableName()).newQuery();
    detailQuery->addToOutputList("BXLUMIVALUE");
    detailQuery->addToOutputList("BXLUMIERROR");
    detailQuery->addToOutputList("BXLUMIQUALITY");
    detailQuery->addToOutputList("ALGONAME");
    detailQuery->setCondition("LUMISUMMARY_ID =:lumisummary_id",detailBindVariables);
    detailQuery->defineOutput(detailOutput);
    coral::ICursor& detailcursor=detailQuery->execute();
    unsigned int nValues=0;
    std::map< std::string,std::vector<float> > bxvaluemap;
    std::map< std::string,std::vector<float> > bxerrormap;
    std::map< std::string,std::vector<short> > bxqualitymap;	
    s=0;
    while( detailcursor.next() ){
      const coral::AttributeList& row=detailcursor.currentRow();     
      std::string algoname=row["algoname"].data<std::string>();
      const coral::Blob& bxlumivalueBlob=row["bxlumivalue"].data<coral::Blob>();
      const void* bxlumivalueStartAddress=bxlumivalueBlob.startingAddress();
      nValues=bxlumivalueBlob.size()/sizeof(float);
      float* bxvalue=new float[lumi::N_BX];
      std::memmove(bxvalue,bxlumivalueStartAddress,bxlumivalueBlob.size());
      bxvaluemap.insert(std::make_pair(algoname,std::vector<float>(bxvalue,bxvalue+nValues)));
      delete [] bxvalue;
      const coral::Blob& bxlumierrorBlob=row["bxlumierror"].data<coral::Blob>();
      const void* bxlumierrorStartAddress=bxlumierrorBlob.startingAddress();
      float* bxerror=new float[lumi::N_BX];
      nValues=bxlumierrorBlob.size()/sizeof(float);
      std::memmove(bxerror,bxlumierrorStartAddress,bxlumierrorBlob.size());
      bxerrormap.insert(std::make_pair(algoname,std::vector<float>(bxerror,bxerror+nValues)));
      delete [] bxerror;
      
      short* bxquality=new short[lumi::N_BX];
      const coral::Blob& bxlumiqualityBlob=row["bxlumiquality"].data<coral::Blob>();
      const void* bxlumiqualityStartAddress=bxlumiqualityBlob.startingAddress();
      nValues=bxlumiqualityBlob.size()/sizeof(short);
      std::memmove(bxquality,bxlumiqualityStartAddress,bxlumiqualityBlob.size());
      bxqualitymap.insert(std::make_pair(algoname,std::vector<short>(bxquality,bxquality+nValues)));
      delete [] bxquality;
      ++s;
    }
    delete detailQuery;
    //for( std::map<std::string,std::vector<float> >::iterator it=bxerrormap.begin(); it!=bxerrormap.end();++it){
      //std::cout<<"algo name "<<it->first<<std::endl;
      //std::cout<<"errorsize "<<(it->second).size()<<std::endl;
      //std::cout<<"first error value "<<*((it->second).begin())<<std::endl;
    //}
    pIn2->setLumiVersion(m_lumiversion);
    if(s!=0){
      pIn2->swapValueData(bxvaluemap);
      pIn2->swapErrorData(bxerrormap);
      pIn2->swapQualData(bxqualitymap);
    }
    //
    //select trgcount,deadtime,prescale,bitname from TRG where runnum=:runnumber  AND cmslsnum=:cmslsnum order by cmslsnum,bitnum; 
    //
    coral::AttributeList trgBindVariables;
    trgBindVariables.extend("runnumber",typeid(unsigned int));
    trgBindVariables.extend("cmslsnum",typeid(unsigned int));

    trgBindVariables["runnumber"].data<unsigned int>()=runnumber;
    trgBindVariables["cmslsnum"].data<unsigned int>()=luminum;

    coral::AttributeList trgOutput;
    trgOutput.extend("trgcount",typeid(unsigned int));
    trgOutput.extend("deadtime",typeid(unsigned long long));
    trgOutput.extend("prescale",typeid(unsigned int));
    trgOutput.extend("bitname",typeid(std::string));
    
    coral::IQuery* trgQuery=schema.tableHandle(lumi::LumiNames::trgTableName()).newQuery();
    trgQuery->addToOutputList("TRGCOUNT");
    trgQuery->addToOutputList("DEADTIME");
    trgQuery->addToOutputList("PRESCALE");
    trgQuery->addToOutputList("BITNAME");
    trgQuery->setCondition("RUNNUM =:runnumber AND CMSLSNUM =:cmslsnum",trgBindVariables);
    trgQuery->addToOrderList("CMSLSNUM");
    trgQuery->addToOrderList("BITNUM");
    trgQuery->defineOutput(trgOutput);
    coral::ICursor& trgcursor=trgQuery->execute();
    unsigned long long deadtime=0;
    //unsigned int trgcount,prescale;
    std::string bitname;
    std::vector< LumiSummary::L1 > trgdata;
    while( trgcursor.next() ){
      LumiSummary::L1 l1;
      const coral::AttributeList& row=trgcursor.currentRow();     
      //row.toOutputStream( std::cout ) << std::endl;
      deadtime=row["deadtime"].data<unsigned long long>();
      l1.ratecount=row["trgcount"].data<unsigned int>();
      l1.prescale=row["prescale"].data<unsigned int>();
      l1.triggername=row["bitname"].data<std::string>();
      trgdata.push_back(l1);
      //std::cout<<"deadtime : "<<deadtime<<", trgcount : "<<trgcount<<", prescale : "<<prescale<<",bitname : "<<bitname<<std::endl;
    }
    pIn1->setDeadtime(deadtime);
    pIn1->swapL1Data(trgdata);
    delete trgQuery;
    
    //
    //select pathname,inputcount,acceptcount,prescale from HLT where runnum=:runnumber  AND cmslsnum=:cmslsnum order by cmslsnum;
    //
    coral::AttributeList hltBindVariables;
    hltBindVariables.extend("runnumber",typeid(unsigned int));
    hltBindVariables.extend("cmslsnum",typeid(unsigned int));

    hltBindVariables["runnumber"].data<unsigned int>()=runnumber;
    hltBindVariables["cmslsnum"].data<unsigned int>()=luminum;

    coral::AttributeList hltOutput;
    hltOutput.extend("pathname",typeid(std::string));
    hltOutput.extend("inputcount",typeid(unsigned int));
    hltOutput.extend("acceptcount",typeid(unsigned int));
    hltOutput.extend("prescale",typeid(unsigned int));
    
    coral::IQuery* hltQuery=schema.tableHandle(lumi::LumiNames::hltTableName()).newQuery();
    hltQuery->addToOutputList("PATHNAME");
    hltQuery->addToOutputList("INPUTCOUNT");
    hltQuery->addToOutputList("ACCEPTCOUNT");
    hltQuery->addToOutputList("PRESCALE");
    hltQuery->setCondition("RUNNUM =:runnumber AND CMSLSNUM =:cmslsnum",hltBindVariables);
    hltQuery->addToOrderList("CMSLSNUM");
    hltQuery->defineOutput(hltOutput);
    coral::ICursor& hltcursor=hltQuery->execute();
    //std::string hltpathname;
    //unsigned int hltinputcount,hltacceptcount,hltprescale;
    std::vector< LumiSummary::HLT > hltdata;
    while( hltcursor.next() ){
      LumiSummary::HLT hlt;
      const coral::AttributeList& row=hltcursor.currentRow();     
      //row.toOutputStream( std::cout ) << std::endl;
      hlt.pathname=row["pathname"].data<std::string>();
      hlt.inputcount=row["inputcount"].data<unsigned int>();
      hlt.ratecount=row["acceptcount"].data<unsigned int>();
      hlt.prescale=row["prescale"].data<unsigned int>();
      hltdata.push_back(hlt);
      //std::cout<<"hltpath : "<<hltpathname<<", inputcount : "<<hltinputcount<<", acceptcount : "<<hltacceptcount<<", prescale : "<<hltprescale<<std::endl;
    }
    pIn1->swapHLTData(hltdata);
    delete hltQuery;

    pOut1.reset(pIn1);
    iLBlock.put(pOut1);

    pOut2.reset(pIn2);
    iLBlock.put(pOut2);
    
  }catch( const coral::Exception& er){
    session->transaction().rollback();
    mydbservice->disconnect(session);
    throw er;
  }
  session->transaction().commit();
  mydbservice->disconnect(session);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(LumiProducer);

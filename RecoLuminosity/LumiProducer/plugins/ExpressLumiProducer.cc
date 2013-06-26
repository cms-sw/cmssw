// -*- C++ -*-
//
// Package:    LumiProducer
// Class:      ExpressLumiProducer
// 
/**\class ExpressLumiProducer ExpressLumiProducer.cc RecoLuminosity/LumiProducer/src/ExpressLumiProducer.cc
Description: A essource/esproducer for lumi values from DIP via runtime logger DB
*/
// read lumi from dip database and dump to express stream
// $Id: ExpressLumiProducer.cc,v 1.4 2013/05/17 20:54:13 chrjones Exp $

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

#include "RecoLuminosity/LumiProducer/interface/DBService.h"
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
#include "boost/filesystem/path.hpp"
#include "boost/filesystem/operations.hpp"

namespace edm {
  class EventSetup;
}

//
// class declaration
//
class ExpressLumiProducer : public edm::one::EDProducer<edm::BeginLuminosityBlockProducer> {
public:
  struct PerLSData{
    unsigned int lsnum;
    float lumivalue;
    unsigned long long deadcount;
    unsigned int numorbit;
    unsigned int startorbit;
    unsigned int bitzerocount;
    std::vector<float> bunchlumivalue;
    std::vector<float> bunchlumierror;
    std::vector<short> bunchlumiquality;
  };
  
  explicit ExpressLumiProducer(const edm::ParameterSet&);
  
  ~ExpressLumiProducer();
  
private:
  

  virtual void produce(edm::Event&, const edm::EventSetup&) override final;

  virtual void beginLuminosityBlockProduce(edm::LuminosityBlock & iLBlock,
				    edm::EventSetup const& iSetup) override final;

  bool fillLumi(edm::LuminosityBlock & iLBlock);
  void fillLSCache(unsigned int runnum,unsigned int luminum);
  void writeProductsForEntry(edm::LuminosityBlock & iLBlock,unsigned int luminum);
  void writeEmptyProductForEntry(edm::LuminosityBlock &iLBlock);
  unsigned int maxavailableLSforRun(coral::ISchema& schema,const std::string&tablename,unsigned int runnumber);
  std::string m_connectStr;
  unsigned int m_cachedrun;
  bool m_isNullRun; //if lumi data exist for this run
  unsigned int m_cachesize;
  std::map< unsigned int,PerLSData > m_lscache;
};

ExpressLumiProducer::
ExpressLumiProducer::ExpressLumiProducer(const edm::ParameterSet& iConfig):m_cachedrun(0),m_isNullRun(false),m_cachesize(0)
{
  // register your products
  produces<LumiSummary, edm::InLumi>();
  produces<LumiDetails, edm::InLumi>();
  // set up cache
  m_connectStr=iConfig.getParameter<std::string>("connect");
  m_cachesize=iConfig.getUntrackedParameter<unsigned int>("ncacheEntries",5);
}

ExpressLumiProducer::~ExpressLumiProducer(){ 
}

//
// member functions
//
void 
ExpressLumiProducer::produce(edm::Event& e, const edm::EventSetup& iSetup)
{ 
}

void 
ExpressLumiProducer::writeEmptyProductForEntry(edm::LuminosityBlock &iLBlock){
  std::auto_ptr<LumiSummary> pOut1;
  std::auto_ptr<LumiDetails> pOut2;
  LumiSummary* pIn1=new LumiSummary;
  LumiDetails* pIn2=new LumiDetails;
  pOut1.reset(pIn1);
  iLBlock.put(pOut1);
  pOut2.reset(pIn2);
  iLBlock.put(pOut2);
}
void 
ExpressLumiProducer::beginLuminosityBlockProduce(edm::LuminosityBlock &iLBlock, edm::EventSetup const &iSetup)
{
  unsigned int currentrun=iLBlock.run();
  unsigned int currentls=iLBlock.luminosityBlock();
  //if is null run, fill empty values and return
  if(m_isNullRun){
    writeEmptyProductForEntry(iLBlock);
    return;
  }
  if(m_cachedrun!=currentrun){
    fillLSCache(currentrun,currentls);
  }else{
    if(m_lscache.find(currentls)==m_lscache.end()){
      //if runnumber is cached but LS is not, this is the first LS, fill LS cache to full capacity
      fillLSCache(currentrun,currentls);
    }
  }
  if( m_lscache.empty() ){
    writeEmptyProductForEntry(iLBlock);
    return;
  }
  unsigned int lstowriteout=0;
  if(m_lscache.find(currentls)==m_lscache.end()){//if the currentls is not in the cache
    std::vector<unsigned int> v;
    for(std::map<unsigned int,ExpressLumiProducer::PerLSData >::iterator it=m_lscache.begin();it!=m_lscache.end();++it){
      v.push_back(it->first);
    }
    lstowriteout=v.back();//last available
  }else{//if the current ls is cached
    lstowriteout=currentls;
  }
  //here the presence of ls is guaranteed
  //std::cout<<"writing "<<runnumber<<" "<<luminum<<std::endl;
  if(lstowriteout==0){
    writeEmptyProductForEntry(iLBlock);
  }else{
    writeProductsForEntry(iLBlock,lstowriteout); 
  }
}

unsigned int
ExpressLumiProducer::maxavailableLSforRun(coral::ISchema& schema,const std::string&tablename,unsigned int runnumber){
  /**
     select  max(lumisection) as maxavailable from  cms_runtime_logger.lumi_sections where runnumber=:runnumber;
  **/
  unsigned int result=0;
  coral::AttributeList bindVariables;
  bindVariables.extend("runnumber",typeid(unsigned int));
  bindVariables["runnumber"].data<unsigned int>()=runnumber;
  std::string conditionStr("RUNNUMBER=:runnumber");
  coral::AttributeList MyOutput;
  MyOutput.extend("maxavailablels",typeid(unsigned int));
  coral::IQuery* myQuery=schema.newQuery();
  myQuery->addToTableList(tablename);
  myQuery->addToOutputList("max(LUMISECTION)","maxavailablels");
  myQuery->setCondition(conditionStr,bindVariables);
  myQuery->defineOutput(MyOutput);
  coral::ICursor& mycursor=myQuery->execute();
  while( mycursor.next() ){
    const coral::AttributeList& row=mycursor.currentRow();
    if(!row["maxavailablels"].isNull()){
      result=row["maxavailablels"].data<unsigned int>();
    }
  }
  return result;
}
void
ExpressLumiProducer::fillLSCache(unsigned int runnumber,unsigned int currentlsnum){
  m_lscache.clear();
  m_cachedrun=runnumber;
  //
  //queries once per cache refill
  //
  //select lumisection,instlumi,delivlumi,livelumi from cms_runtime_logger.lumi_sections where lumisection>=:lsmin and lumisection<:lsmax and runnumber=:runnumber;
  //
  edm::Service<lumi::service::DBService> mydbservice;
  if( !mydbservice.isAvailable() ){
    throw cms::Exception("Non existing service lumi::service::DBService");
  }
  coral::ISessionProxy* session=mydbservice->connectReadOnly(m_connectStr);
  coral::ITypeConverter& tconverter=session->typeConverter();
  tconverter.setCppTypeForSqlType(std::string("float"),std::string("FLOAT(63)"));
  tconverter.setCppTypeForSqlType(std::string("unsigned int"),std::string("NUMBER(10)"));
  tconverter.setCppTypeForSqlType(std::string("unsigned short"),std::string("NUMBER(1)"));
  unsigned int lsmin=1;
  unsigned int lsmax=currentlsnum;
  try{
    session->transaction().start(true);
    coral::ISchema& schema=session->nominalSchema();
    unsigned int maxavailableLS=maxavailableLSforRun(schema,std::string("LUMI_SECTIONS"),m_cachedrun);
    if(maxavailableLS!=0 && maxavailableLS<currentlsnum){
      lsmax=maxavailableLS;
    }else if(maxavailableLS==0){
      //this run not existing (yet)
      session->transaction().commit();
      mydbservice->disconnect(session);
      return;
    }
    if(m_cachesize!=0){
      lsmin=(lsmax-m_cachesize)>0 ? (lsmax-m_cachesize+1) : 1;
    }
    for(unsigned int n=lsmin;n<=lsmax;++n){
      PerLSData l;
      std::vector<float> mytmp(3564,0.0);
      l.bunchlumivalue.swap(mytmp);
      std::vector<float> myerrtmp(3564,0.0);
      l.bunchlumierror.swap(myerrtmp);
      std::vector<short> myqtmp(3564,0);
      l.bunchlumiquality.swap(myqtmp);
      m_lscache.insert(std::make_pair(n,l));
    }
 
    coral::AttributeList lumisummaryBindVariables;
    lumisummaryBindVariables.extend("lsmin",typeid(unsigned int));
    lumisummaryBindVariables.extend("runnumber",typeid(unsigned int));
    lumisummaryBindVariables["runnumber"].data<unsigned int>()=m_cachedrun;
    lumisummaryBindVariables["lsmin"].data<unsigned int>()=lsmin;
    std::string conditionStr(" RUNNUMBER=:runnumber AND LUMISECTION>=:lsmin ");
    coral::AttributeList lumisummaryOutput;
    lumisummaryOutput.extend("LUMISECTION",typeid(unsigned int));
    lumisummaryOutput.extend("INSTLUMI",typeid(float));
    lumisummaryOutput.extend("DELIVLUMISECTION",typeid(float));
    lumisummaryOutput.extend("LIVELUMISECTION",typeid(float));
    lumisummaryOutput.extend("STARTORBIT",typeid(unsigned long long));
    if(m_cachesize!=0){
      lumisummaryBindVariables.extend("lsmax",typeid(unsigned int));
      conditionStr=conditionStr+"AND LUMISECTION<=:lsmax";
      lumisummaryBindVariables["lsmax"].data<unsigned int>()=lsmax;      
    }
    coral::IQuery* lumisummaryQuery=schema.newQuery();
    lumisummaryQuery->addToTableList(std::string("LUMI_SECTIONS"));
    lumisummaryQuery->addToOutputList("LUMISECTION");
    lumisummaryQuery->addToOutputList("INSTLUMI");
    lumisummaryQuery->addToOutputList("DELIVLUMISECTION");
    lumisummaryQuery->addToOutputList("LIVELUMISECTION");
    lumisummaryQuery->addToOutputList("STARTORBIT");
    lumisummaryQuery->setCondition(conditionStr,lumisummaryBindVariables);
    lumisummaryQuery->defineOutput(lumisummaryOutput);
    coral::ICursor& lumisummarycursor=lumisummaryQuery->execute();
    unsigned int rowcounter=0;
    while( lumisummarycursor.next() ){
      const coral::AttributeList& row=lumisummarycursor.currentRow();
      unsigned int lsnum=row["LUMISECTION"].data<unsigned int>();
      float instlumi=0.0;
      if(!row["INSTLUMI"].isNull()){
	instlumi=row["INSTLUMI"].data<float>();//Hz/ub
      }
      float deadfrac=1.0;
      float intgdellumi=0.0;
      float intgreclumi=0.0;
      unsigned long long startorbit=0;
      if(!row["DELIVLUMISECTION"].isNull()){
	intgdellumi=row["DELIVLUMISECTION"].data<float>()*1000.0;//convert to /ub
      }
      if(!row["LIVELUMISECTION"].isNull()){
	intgreclumi=row["LIVELUMISECTION"].data<float>()*1000.0;//convert to /ub
      }
      if(intgdellumi>0){
	deadfrac=1.0-intgreclumi/intgdellumi;
      }
      if(!row["STARTORBIT"].isNull()){
	startorbit=row["STARTORBIT"].data<unsigned long long>();//convert to /ub
      }
      unsigned long long deadcount=deadfrac*10000.0;
      unsigned long long bitzerocount=10000.0;
      PerLSData& lsdata=m_lscache[lsnum];
      lsdata.lsnum=lsnum;
      lsdata.lumivalue=instlumi; 
      lsdata.deadcount=deadcount;
      lsdata.bitzerocount=bitzerocount;
      lsdata.startorbit=startorbit;
      lsdata.numorbit=262144;
      ++rowcounter;
    }
    if (rowcounter==0){
      m_isNullRun=true;
    }
    delete lumisummaryQuery;
    if(m_isNullRun) return;
    //
    //queries once per cache refill
    //
    //select lumisection,bunch,bunchlumi from cms_runtime_logger.bunch_lumi_sections where lumisection>=:lsmin and lumisection<:lsmax and runnumber=:runnumber;
    //
    coral::AttributeList lumidetailBindVariables;
    lumidetailBindVariables.extend("lsmin",typeid(unsigned int));
    lumidetailBindVariables.extend("runnumber",typeid(unsigned int));
    lumidetailBindVariables["runnumber"].data<unsigned int>()=m_cachedrun;
    lumidetailBindVariables["lsmin"].data<unsigned int>()=lsmin;
    std::string detailconditionStr(" RUNNUMBER=:runnumber AND LUMISECTION>=:lsmin AND BUNCHLUMI>0 ");
    coral::AttributeList lumidetailOutput;
    lumidetailOutput.extend("LUMISECTION",typeid(unsigned int));
    lumidetailOutput.extend("BUNCH",typeid(unsigned int));
    lumidetailOutput.extend("BUNCHLUMI",typeid(float));
    if(m_cachesize!=0){
      lumidetailBindVariables.extend("lsmax",typeid(unsigned int));
      detailconditionStr=detailconditionStr+"AND LUMISECTION<=:lsmax";
      lumidetailBindVariables["lsmax"].data<unsigned int>()=lsmax;      
    }
    coral::IQuery* lumidetailQuery=schema.newQuery();
    lumidetailQuery->addToTableList(std::string("BUNCH_LUMI_SECTIONS"));
    lumidetailQuery->addToOutputList("LUMISECTION");
    lumidetailQuery->addToOutputList("BUNCH");
    lumidetailQuery->addToOutputList("BUNCHLUMI");
    lumidetailQuery->setCondition(detailconditionStr,lumidetailBindVariables);
    lumidetailQuery->defineOutput(lumidetailOutput);
    coral::ICursor& lumidetailcursor=lumidetailQuery->execute();
    while( lumidetailcursor.next() ){
      const coral::AttributeList& row=lumidetailcursor.currentRow();
      unsigned int lsnum=row["LUMISECTION"].data<unsigned int>();
      unsigned int bxidx=row["BUNCH"].data<unsigned int>();
      float bxlumi=row["BUNCHLUMI"].data<float>();//Hz/ub
      m_lscache[lsnum].bunchlumivalue[bxidx]=bxlumi;
    }
    delete lumidetailQuery;
    session->transaction().commit();
  }catch(const coral::Exception& er){
    session->transaction().rollback();
    mydbservice->disconnect(session);
    throw cms::Exception("DatabaseError ")<<er.what();
  }
  mydbservice->disconnect(session);
}
void
ExpressLumiProducer::writeProductsForEntry(edm::LuminosityBlock & iLBlock,unsigned int luminum){
  //std::cout<<"writing runnumber,luminum "<<runnumber<<" "<<luminum<<std::endl;
  std::auto_ptr<LumiSummary> pOut1;
  std::auto_ptr<LumiDetails> pOut2;
  LumiSummary* pIn1=new LumiSummary;
  LumiDetails* pIn2=new LumiDetails;
  if(m_isNullRun){
    pIn1->setLumiVersion("DIP");
    pIn2->setLumiVersion("DIP");
    pOut1.reset(pIn1);
    iLBlock.put(pOut1);
    pOut2.reset(pIn2);
    iLBlock.put(pOut2);
    return;
  }
  PerLSData& lsdata=m_lscache[luminum];
  pIn1->setLumiVersion("DIP");
  pIn1->setLumiData(lsdata.lumivalue,0.0,0.0);
  pIn1->setDeadCount(lsdata.deadcount);
  pIn1->setBitZeroCount(lsdata.bitzerocount);
  pIn1->setlsnumber(lsdata.lsnum);
  pIn1->setOrbitData(lsdata.startorbit,lsdata.numorbit);

  pIn2->setLumiVersion("DIP");
  pIn2->fill(LumiDetails::kOCC1,lsdata.bunchlumivalue,lsdata.bunchlumierror,lsdata.bunchlumiquality);
  pOut1.reset(pIn1);
  iLBlock.put(pOut1);
  pOut2.reset(pIn2);
  iLBlock.put(pOut2);
}
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ExpressLumiProducer);

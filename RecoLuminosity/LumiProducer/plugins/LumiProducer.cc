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
// $Id: LumiProducer.cc,v 1.2 2010/03/22 17:29:27 xiezhen Exp $

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

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
  
  virtual void beginLuminosityBlock(edm::LuminosityBlock & iLBlock,
				    edm::EventSetup const& iSetup);
  virtual void endLuminosityBlock(edm::LuminosityBlock& lumiBlock, 
				  edm::EventSetup const& c);
  //void fillDefaultLumi(edm::LuminosityBlock & iLBlock);
  bool fillLumi(edm::LuminosityBlock & iLBlock);

  //edm::ParameterSet pset_;

  std::string m_connectStr;
  std::string m_lumiversion;
};

//
// constructors and destructor
//
LumiProducer::LumiProducer(const edm::ParameterSet& iConfig)
{
  // register your products
  produces<LumiSummary, edm::InLumi>();
  produces<LumiDetails, edm::InLumi>();
  m_connectStr=iConfig.getParameter<std::string>("connect");
  m_lumiversion=iConfig.getUntrackedParameter<std::string>("lumiversion","0001");
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
void LumiProducer::beginLuminosityBlock(edm::LuminosityBlock &iLBlock, edm::EventSetup const &iSetup) {  
}
void LumiProducer::endLuminosityBlock(edm::LuminosityBlock & iLBlock, 
				     edm::EventSetup const& c){
  unsigned int runnumber=iLBlock.run();
  unsigned int luminum=iLBlock.luminosityBlock();
  edm::Service<lumi::service::DBService> mydbservice;
  if( !mydbservice.isAvailable() ){
    std::cout<<"Service is unavailable"<<std::endl;
    return;
  }
  coral::ISessionProxy* session=mydbservice->connectReadOnly(m_connectStr);
  coral::ITypeConverter& tpc=session->typeConverter();
  tpc.setCppTypeForSqlType("unsigned int","NUMBER(7)");
  tpc.setCppTypeForSqlType("unsigned int","NUMBER(10)");
  tpc.setCppTypeForSqlType("unsigned long long","NUMBER(20)");
  try{
    session->transaction().start(true);
    coral::ISchema& schema=session->nominalSchema();
    if(!schema.existsTable(lumi::LumiNames::lumisummaryTableName())){
      throw lumi::Exception(std::string("non-existing table ")+lumi::LumiNames::lumisummaryTableName(),"endLuminosityBlock","LumiProducer");
    }
    if(!schema.existsTable(lumi::LumiNames::lumidetailTableName())){
      throw lumi::Exception(std::string("non-existing table ")+lumi::LumiNames::lumisummaryTableName(),"endLuminosityBlock","LumiProducer");
    }
    if(!schema.existsTable(lumi::LumiNames::trgTableName())){
      throw lumi::Exception(std::string("non-existing table ")+lumi::LumiNames::trgTableName(),"endLuminosityBlock","LumiProducer");
    }
    if(!schema.existsTable(lumi::LumiNames::hltTableName())){
      throw lumi::Exception(std::string("non-existing table ")+lumi::LumiNames::hltTableName(),"endLuminosityBlock","LumiProducer");
    }
    //
    //select cmslsnum,lumisummary_id,instlumi,instlumierror,lumisectionquality,startorbit,numorbit from LUMISUMMARY where runnum=:runnumber AND lumiversion=:lumiversion AND cmslsnum between :cmslsnum and :cmslsnum+4 order by cmslsnum
    //
    coral::AttributeList lumiBindVariables;
    lumiBindVariables.extend("runnumber",typeid(unsigned int));
    lumiBindVariables.extend("cmslsnum",typeid(unsigned int));
    lumiBindVariables.extend("lumiversion",typeid(std::string));

    lumiBindVariables["runnumber"].data<unsigned int>()=runnumber;
    lumiBindVariables["cmslsnum"].data<unsigned int>()=luminum;
    lumiBindVariables["lumiversion"].data<std::string>()=m_lumiversion;
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
    lumiQuery->setCondition("RUNNUM =:runnumber AND LUMIVERSION=:lumiversion AND CMSLSNUM =:cmslsnum",lumiBindVariables);
    lumiQuery->defineOutput(lumiOutput);
    coral::ICursor& lumicursor=lumiQuery->execute();
    unsigned long long lumisummary_id=0;
    float instlumi=0.0;
    float instlumierror=0.0;
    short lumisectionquality=0;
    unsigned int startorbit, numorbit;
    
    while( lumicursor.next() ){
      const coral::AttributeList& row=lumicursor.currentRow();     
      //row.toOutputStream( std::cout ) << std::endl;
      lumisummary_id=row["lumisummary_id"].data<unsigned long long>();
      instlumi=row["instlumi"].data<float>();
      instlumierror=row["instlumierror"].data<float>();
      lumisectionquality=row["lumisectionquality"].data<short>();
      startorbit=row["startorbit"].data<unsigned int>();
      numorbit=row["numorbit"].data<unsigned int>();
    }
    delete lumiQuery;

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
    while( detailcursor.next() ){
      const coral::AttributeList& row=detailcursor.currentRow();     
      std::string algoname=row["algoname"].data<std::string>();
      const coral::Blob& bxlumivalueBlob=row["bxlumivalue"].data<coral::Blob>();
      const void* bxlumivalueStartAddress=bxlumivalueBlob.startingAddress();
      nValues=bxlumivalueBlob.size()/sizeof(float);
      float* bxvalue=new float[lumi::N_BX];
      std::memmove(bxvalue,bxlumivalueStartAddress,nValues);
      bxvaluemap.insert(std::make_pair(algoname,std::vector<float>(bxvalue,bxvalue+nValues)));
      delete [] bxvalue;
      const coral::Blob& bxlumierrorBlob=row["bxlumierror"].data<coral::Blob>();
      const void* bxlumierrorStartAddress=bxlumierrorBlob.startingAddress();
      float* bxerror=new float[lumi::N_BX];
      nValues=bxlumierrorBlob.size()/sizeof(float);
      std::memmove(bxerror,bxlumierrorStartAddress,nValues);
      bxerrormap.insert(std::make_pair(algoname,std::vector<float>(bxerror,bxerror+nValues)));
      delete [] bxerror;
      
      short* bxquality=new short[lumi::N_BX];
      const coral::Blob& bxlumiqualityBlob=row["bxlumiquality"].data<coral::Blob>();
      const void* bxlumiqualityStartAddress=bxlumiqualityBlob.startingAddress();
      nValues=bxlumiqualityBlob.size()/sizeof(short);
      std::memmove(bxquality,bxlumiqualityStartAddress,nValues);
      bxqualitymap.insert(std::make_pair(algoname,std::vector<short>(bxquality,bxquality+nValues)));
      delete [] bxquality;
    }
    delete detailQuery;
    /**
       for( std::map<std::string,std::vector<float> >::iterator it=bxerrormap.begin(); it!=bxerrormap.end();++it){
       std::cout<<"algo name "<<it->first<<std::endl;
       std::cout<<"errorsize "<<(it->second).size()<<std::endl;
       std::cout<<"first error value "<<*((it->second).begin())<<std::endl;
       }
    **/
    //
    //select trgcount,deadtime,prescale,bitname from TRG where runnum=:runnumber  AND cmslsnum between :lsnum and :lsnum+4 order by cmslsnum,bitnum; 
    //
    coral::AttributeList trgBindVariables;
    trgBindVariables.extend("runnumber",typeid(unsigned int));
    trgBindVariables.extend("cmslsnum",typeid(unsigned int));

    trgBindVariables["runnumber"].data<unsigned int>()=runnumber;
    trgBindVariables["cmslsnum"].data<unsigned int>()=luminum;

    coral::AttributeList trgOutput;
    trgOutput.extend("count",typeid(unsigned int));
    trgOutput.extend("deadtime",typeid(unsigned long long));
    trgOutput.extend("prescale",typeid(unsigned int));
    trgOutput.extend("bitname",typeid(std::string));
    
    coral::IQuery* trgQuery=schema.tableHandle(lumi::LumiNames::trgTableName()).newQuery();
    trgQuery->addToOutputList("COUNT");
    trgQuery->addToOutputList("DEADTIME");
    trgQuery->addToOutputList("PRESCALE");
    trgQuery->addToOutputList("BITNAME");
    trgQuery->setCondition("RUNNUM =:runnumber AND CMSLSNUM =:cmslsnum",trgBindVariables);
    trgQuery->addToOrderList("CMSLSNUM");
    trgQuery->addToOrderList("BITNUM");
    trgQuery->defineOutput(trgOutput);
    coral::ICursor& trgcursor=trgQuery->execute();
    unsigned long long deadtime=0;
    unsigned int trgcount,prescale;
    std::string bitname;
    while( trgcursor.next() ){
      const coral::AttributeList& row=trgcursor.currentRow();     
      //row.toOutputStream( std::cout ) << std::endl;
      deadtime=row["deadtime"].data<unsigned long long>();
      trgcount=row["count"].data<unsigned int>();
      prescale=row["prescale"].data<unsigned int>();
      bitname=row["bitname"].data<std::string>();
      std::cout<<"deadtime : "<<deadtime<<", trgcount : "<<trgcount<<", prescale : "<<prescale<<",bitname : "<<bitname<<std::endl;
    }
    delete trgQuery;
    
    //
    //select pathname,inputcount,acceptcount,prescale from HLT where runnum=:runnumber  AND cmslsnum between :lsnum and :lsnum+4 order by cmslsnum;
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
    std::string hltpathname;
    unsigned int hltinputcount,hltacceptcount,hltprescale;
    while( hltcursor.next() ){
      const coral::AttributeList& row=hltcursor.currentRow();     
      //row.toOutputStream( std::cout ) << std::endl;
      hltpathname=row["pathname"].data<std::string>();
      hltinputcount=row["inputcount"].data<unsigned int>();
      hltacceptcount=row["acceptcount"].data<unsigned int>();
      hltprescale=row["prescale"].data<unsigned int>();
      std::cout<<"hltpath : "<<hltpathname<<", inputcount : "<<hltinputcount<<", acceptcount : "<<hltacceptcount<<", prescale : "<<hltprescale<<std::endl;
    }
    delete hltQuery;

    std::auto_ptr<LumiSummary> pOut1;
    LumiSummary* pIn1=new LumiSummary;
    
    pOut1.reset(pIn1);
    iLBlock.put(pOut1);

    std::auto_ptr<LumiDetails> pOut2;
    LumiDetails* pIn2=new LumiDetails;
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

#ifndef RecoLuminosity_LumiProducer_TRGWBM2DB_h 
#define RecoLuminosity_LumiProducer_TRGWBM2DB_h 
#include "CoralBase/AttributeList.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/AttributeSpecification.h"
#include "CoralBase/Exception.h"
#include "RelationalAccess/ConnectionService.h"
#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/ITransaction.h"
#include "RelationalAccess/ITypeConverter.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ICursor.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/IView.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/ITableDataEditor.h"
#include "RelationalAccess/IBulkOperation.h"
#include "RecoLuminosity/LumiProducer/interface/DataPipe.h"
#include "RecoLuminosity/LumiProducer/interface/LumiNames.h"
#include "RecoLuminosity/LumiProducer/interface/idDealer.h"
#include "RecoLuminosity/LumiProducer/interface/Exception.h"
#include "RecoLuminosity/LumiProducer/interface/DBConfig.h"
#include "RecoLuminosity/LumiProducer/interface/ConstantDef.h"
#include <iostream>
#include <sstream>
#include <map>
namespace lumi{
  class TRGWBM2DB : public DataPipe{
  public:
    explicit TRGWBM2DB(const std::string& dest);
    virtual void retrieveData( unsigned int runnumber);
    virtual const std::string dataType() const;
    virtual const std::string sourceType() const;
    virtual ~TRGWBM2DB();
  private:
    std::string int2str(unsigned int t,unsigned int width);
    unsigned int str2int(const std::string& s);
  private:
    //per run information
    typedef std::vector<std::string> TriggerNameResult_Algo;
    typedef std::vector<std::string> TriggerNameResult_Tech;
    typedef std::vector<unsigned int> PrescaleResult_Algo;
    typedef std::vector<unsigned int> PrescaleResult_Tech;
    //per lumisection information
    typedef unsigned long long DEADCOUNT;
    typedef std::vector<DEADCOUNT> TriggerDeadCountResult;
    typedef std::vector<unsigned int> BITCOUNT;
    typedef std::vector<BITCOUNT> TriggerCountResult_Algo;
    typedef std::vector<BITCOUNT> TriggerCountResult_Tech;
  };//cl TRGWBM2DB
  //
  //implementation
  //
 TRGWBM2DB::TRGWBM2DB(const std::string& dest):DataPipe(dest){}
  void TRGWBM2DB::retrieveData( unsigned int runnumber){
    std::string runnumberstr=int2str(runnumber,6);
    //query source GT database
    coral::ConnectionService* svc=new coral::ConnectionService;
    lumi::DBConfig dbconf(*svc);
    if(!m_authpath.empty()){
      dbconf.setAuthentication(m_authpath);
    }
    //std::cout<<"m_source "<<m_source<<std::endl;
    coral::ISessionProxy* trgsession=svc->connect(m_source, coral::ReadOnly);
    coral::ITypeConverter& tpc=trgsession->typeConverter();

    tpc.setCppTypeForSqlType("unsigned int","NUMBER(7)");
    tpc.setCppTypeForSqlType("unsigned int","NUMBER(10)");
    tpc.setCppTypeForSqlType("unsigned long long","NUMBER(20)");
    
    coral::AttributeList bindVariableList;
    bindVariableList.extend("runnumber",typeid(unsigned int));
    bindVariableList["runnumber"].data<unsigned int>()=runnumber;
    std::string wbmschema("CMS_WBM");
    std::string algoname("LEVEL1_TRIGGER_ALGO_CONDITIONS");
    std::string techname("LEVEL1_TRIGGER_TECH_CONDITIONS");
    std::string deadname("LEVEL1_TRIGGER_CONDITIONS");
    
    std::string gtschema("CMS_GT");
    std::string runtechviewname("GT_RUN_TECH_VIEW");
    std::string runalgoviewname("GT_RUN_ALGO_VIEW");
    std::string runprescalgoviewname("GT_RUN_PRESC_ALGO_VIEW");
    std::string runpresctechviewname("GT_RUN_PRESC_TECH_VIEW");

    //data exchange format
    lumi::TRGWBM2DB::BITCOUNT mybitcount_algo;
    mybitcount_algo.reserve(lumi::N_TRGALGOBIT);
    lumi::TRGWBM2DB::BITCOUNT mybitcount_tech; 
    mybitcount_tech.reserve(lumi::N_TRGTECHBIT);
    lumi::TRGWBM2DB::TriggerNameResult_Algo algonames;
    algonames.reserve(lumi::N_TRGALGOBIT);
    lumi::TRGWBM2DB::TriggerNameResult_Tech technames;
    technames.reserve(lumi::N_TRGTECHBIT);
    lumi::TRGWBM2DB::PrescaleResult_Algo algoprescale;
    algoprescale.reserve(lumi::N_TRGALGOBIT);
    lumi::TRGWBM2DB::PrescaleResult_Tech techprescale;
    techprescale.reserve(lumi::N_TRGTECHBIT);
    lumi::TRGWBM2DB::TriggerCountResult_Algo algocount;
    algocount.reserve(400);
    lumi::TRGWBM2DB::TriggerCountResult_Tech techcount;
    techcount.reserve(400);
    lumi::TRGWBM2DB::TriggerDeadCountResult deadtimeresult;
    deadtimeresult.reserve(400);
    coral::ITransaction& transaction=trgsession->transaction();
    transaction.start(true);
    //uncomment if you want to see all the visible views
    /**
       transaction.start(true); //true means readonly transaction
       std::cout<<"schema name "<<session->schema(gtmonschema).schemaName()<<std::endl;
       std::set<std::string> listofviews;
       listofviews=session->schema(gtmonschema).listViews();
       for( std::set<std::string>::iterator it=listofviews.begin(); it!=listofviews.end();++it ){
       std::cout<<"view: "<<*it<<std::endl;
       } 
       std::cout<<"schema name "<<session->schema(gtschema).schemaName()<<std::endl;
       listofviews.clear();
       listofviews=session->schema(gtschema).listViews();
       for( std::set<std::string>::iterator it=listofviews.begin(); it!=listofviews.end();++it ){
       std::cout<<"view: "<<*it<<std::endl;
       } 
       std::cout<<"commit transaction"<<std::endl;
       transaction.commit();
    **/
    /**
       Part I
       query tables in schema cms_wbm
    **/

    coral::ISchema& wbmschemaHandle=trgsession->schema(wbmschema);    
    if(!wbmschemaHandle.existsTable(algoname)){
      throw lumi::Exception(std::string("non-existing table ")+algoname,"retrieveData","TRGWBM2DB");
    }
    if(!wbmschemaHandle.existsTable(techname)){
      throw lumi::Exception(std::string("non-existing table ")+techname,"retrieveData","TRGWBM2DB");
    }
    if(!wbmschemaHandle.existsTable(deadname)){
      throw lumi::Exception(std::string("non-existing table ")+deadname,"retrieveData","TRGWBM2DB");
    }
    //
    //select LUMISEGMENTNR,GTALGOCOUNTS,BIT from cms_wbm.LEVEL1_TRIGGER_ALGO_CONDITIONS where RUNNUMBER=133881 order by LUMISEGMENTNR,BIT;
    //
    coral::IQuery* Queryalgo=wbmschemaHandle.newQuery();
    Queryalgo->addToTableList(algoname);
    coral::AttributeList qalgoOutput;
    qalgoOutput.extend("counts",typeid(unsigned int));
    qalgoOutput.extend("lsnr",typeid(unsigned int));
    qalgoOutput.extend("algobit",typeid(unsigned int));
    Queryalgo->addToOutputList("GTALGOCOUNTS","counts");
    Queryalgo->addToOutputList("LUMISEGMENTNR","lsnr");
    Queryalgo->addToOutputList("BIT","algobit");
    Queryalgo->setCondition("RUNNUMBER =:runnumber",bindVariableList);
    Queryalgo->addToOrderList("lsnr");
    Queryalgo->addToOrderList("algobit");
    Queryalgo->defineOutput(qalgoOutput);
    coral::ICursor& c=Queryalgo->execute();
    
    unsigned int s=0;
    while( c.next() ){
      const coral::AttributeList& row = c.currentRow();     
      unsigned int count=row["counts"].data<unsigned int>();
      unsigned int algobit=row["algobit"].data<unsigned int>();
      mybitcount_algo.push_back(count);
      if(algobit==(lumi::N_TRGALGOBIT-1)){
	algocount.push_back(mybitcount_algo);
	mybitcount_algo.clear();
      }
      ++s;
    }
    if(s==0){
      c.close();
      delete Queryalgo;
      transaction.commit();
      throw lumi::Exception(std::string("requested run ")+runnumberstr+std::string(" doesn't exist for algocounts"),"retrieveData","TRGWBM2DB");
    }
    delete Queryalgo;
    //
    //select LUMISEGMENTNR,GTTECHCOUNTS,BIT from cms_wbm.LEVEL1_TRIGGER_TECH_CONDITIONS where RUNNUMBER=133881 order by LUMISEGMENTNR,BIT;
    //
    coral::IQuery* Querytech=wbmschemaHandle.newQuery();
    Querytech->addToTableList(techname);
    coral::AttributeList qtechOutput;
    qtechOutput.extend("counts",typeid(unsigned int));
    qtechOutput.extend("lsnr",typeid(unsigned int));
    qtechOutput.extend("techbit",typeid(unsigned int));
    Querytech->addToOutputList("GTTECHCOUNTS","counts");
    Querytech->addToOutputList("LUMISEGMENTNR","lsnr");
    Querytech->addToOutputList("BIT","techbit");
    Querytech->setCondition("RUNNUMBER =:runnumber",bindVariableList);
    Querytech->addToOrderList("lsnr");
    Querytech->addToOrderList("techbit");
    Querytech->defineOutput(qtechOutput);
    coral::ICursor& techcursor=Querytech->execute();
    
    s=0;
    while( techcursor.next() ){
      const coral::AttributeList& row = techcursor.currentRow();     
      //row.toOutputStream( std::cout ) << std::endl;
      //unsigned int lsnr=row["lsnr"].data<unsigned int>();
      unsigned int count=row["counts"].data<unsigned int>();
      unsigned int techbit=row["techbit"].data<unsigned int>();
      mybitcount_tech.push_back(count);
      if(techbit==(lumi::N_TRGTECHBIT-1)){
	techcount.push_back(mybitcount_tech);
	mybitcount_tech.clear();
      }
      ++s;
    }
    if(s==0){
      techcursor.close();
      delete Querytech;
      transaction.commit();
      throw lumi::Exception(std::string("requested run ")+runnumberstr+std::string(" doesn't exist for techcounts"),"retrieveData","TRGWBM2DB");
    }
    delete Querytech;

    //
    //select LUMISEGMENTNR,DEADTIMEBEAMACTIVE from cms_wbm.LEVEL1_TRIGGER_CONDITIONS where RUNNUMBER=133881 order by LUMISEGMENTNR;
    //
    coral::IQuery* Querydead=wbmschemaHandle.newQuery();
    Querydead->addToTableList(deadname);
    coral::AttributeList qdeadOutput;
    qdeadOutput.extend("counts",typeid(unsigned int));
    qdeadOutput.extend("lsnr",typeid(unsigned int));
    Querydead->addToOutputList("DEADTIMEBEAMACTIVE","counts");
    Querydead->addToOutputList("LUMISEGMENTNR","lsnr");
    coral::AttributeList bindVariablesDead;
    bindVariablesDead.extend("runnumber",typeid(int));
    bindVariablesDead["runnumber"].data<int>()=runnumber;
    Querydead->setCondition("LUMISEGMENTNR =:runnumber",bindVariablesDead);
    Querydead->addToOrderList("lsnr");
    Querydead->defineOutput(qdeadOutput);
    coral::ICursor& deadcursor=Querydead->execute();
    s=0;
    while( deadcursor.next() ){
      const coral::AttributeList& row = deadcursor.currentRow();     
      //row.toOutputStream( std::cout ) << std::endl;
      //unsigned int lsnr=row["lsnr"].data<unsigned int>();
      unsigned int count=row["counts"].data<unsigned int>();
      deadtimeresult.push_back(count);
      ++s;
    }
    if(s==0){
      deadcursor.close();
      delete Querydead;
      transaction.commit();
      throw lumi::Exception(std::string("requested run ")+runnumberstr+std::string(" doesn't exist for deadcounts"),"retrieveData","TRGWBM2DB");
      return;
    }
    //transaction.commit();
    delete Querydead;
    /**
       Part II
       query tables in schema cms_gt
    **/
    coral::ISchema& gtschemaHandle=trgsession->schema(gtschema);
    if(!gtschemaHandle.existsView(runtechviewname)){
      throw lumi::Exception(std::string("non-existing view ")+runtechviewname,"str2int","TRGWBM2DB");
    }
    if(!gtschemaHandle.existsView(runalgoviewname)){
      throw lumi::Exception(std::string("non-existing view ")+runalgoviewname,"str2int","TRGWBM2DB");
    }
    if(!gtschemaHandle.existsView(runprescalgoviewname)){
      throw lumi::Exception(std::string("non-existing view ")+runprescalgoviewname,"str2int","TRGWBM2DB");
    }
    if(!gtschemaHandle.existsView(runpresctechviewname)){
      throw lumi::Exception(std::string("non-existing view ")+runpresctechviewname,"str2int","TRGWBM2DB");
    }
    //
    //select algo_index,alias from cms_gt.gt_run_algo_view where runnumber=:runnumber order by algo_index;
    //
    std::map<unsigned int,std::string> triggernamemap;
    coral::IQuery* QueryName=gtschemaHandle.newQuery();
    QueryName->addToTableList(runalgoviewname);
    coral::AttributeList qAlgoNameOutput;
    qAlgoNameOutput.extend("algo_index",typeid(unsigned int));
    qAlgoNameOutput.extend("alias",typeid(std::string));
    QueryName->addToOutputList("algo_index");
    QueryName->addToOutputList("alias");
    QueryName->setCondition("runnumber =:runnumber",bindVariableList);
    QueryName->addToOrderList("algo_index");
    QueryName->defineOutput(qAlgoNameOutput);
    coral::ICursor& algonamecursor=QueryName->execute();
    while( algonamecursor.next() ){
      const coral::AttributeList& row = algonamecursor.currentRow();     
      //row.toOutputStream( std::cout ) << std::endl;
      unsigned int algo_index=row["algo_index"].data<unsigned int>();
      std::string algo_name=row["alias"].data<std::string>();
      triggernamemap.insert(std::make_pair(algo_index,algo_name));
    }
    delete QueryName;
    //
    //select techtrig_index,name from cms_gt.gt_run_tech_view where runnumber=:runnumber order by techtrig_index;
    //
    std::map<unsigned int,std::string> techtriggernamemap;
    coral::IQuery* QueryTechName=gtschemaHandle.newQuery();
    QueryTechName->addToTableList(runtechviewname);
    coral::AttributeList qTechNameOutput;
    qTechNameOutput.extend("techtrig_index",typeid(unsigned int));
    qTechNameOutput.extend("name",typeid(std::string));
    QueryTechName->addToOutputList("techtrig_index");
    QueryTechName->addToOutputList("name");
    QueryTechName->setCondition("runnumber =:runnumber",bindVariableList);
    QueryTechName->addToOrderList("techtrig_index");
    QueryTechName->defineOutput(qTechNameOutput);
    coral::ICursor& technamecursor=QueryTechName->execute();
    while( technamecursor.next() ){
      const coral::AttributeList& row = technamecursor.currentRow();     
      //row.toOutputStream( std::cout ) << std::endl;
      unsigned int tech_index=row["techtrig_index"].data<unsigned int>();
      std::string tech_name=row["name"].data<std::string>();
      techtriggernamemap.insert(std::make_pair(tech_index,tech_name));
    }
    delete QueryTechName;
    //
    //select prescale_factor_algo_000,prescale_factor_algo_001..._127 from cms_gt.gt_run_presc_algo_view where runr=:runnumber and prescale_index=0;
    //    
    coral::IQuery* QueryAlgoPresc=gtschemaHandle.newQuery();
    QueryAlgoPresc->addToTableList(runprescalgoviewname);
    coral::AttributeList qAlgoPrescOutput;
    std::string algoprescBase("PRESCALE_FACTOR_ALGO_");
    for(unsigned int bitidx=0;bitidx<lumi::N_TRGALGOBIT;++bitidx){
      std::string algopresc=algoprescBase+int2str(bitidx,3);
      qAlgoPrescOutput.extend(algopresc,typeid(unsigned int));
    }
    for(unsigned int bitidx=0;bitidx<lumi::N_TRGALGOBIT;++bitidx){
      std::string algopresc=algoprescBase+int2str(bitidx,3);
      QueryAlgoPresc->addToOutputList(algopresc);
    }
    coral::AttributeList PrescbindVariable;
    PrescbindVariable.extend("runnumber",typeid(int));
    PrescbindVariable.extend("prescaleindex",typeid(int));
    PrescbindVariable["runnumber"].data<int>()=runnumber;
    PrescbindVariable["prescaleindex"].data<int>()=0;
    QueryAlgoPresc->setCondition("runnr =:runnumber AND prescale_index =:prescaleindex",PrescbindVariable);
    QueryAlgoPresc->defineOutput(qAlgoPrescOutput);
    coral::ICursor& algopresccursor=QueryAlgoPresc->execute();
    while( algopresccursor.next() ){
      const coral::AttributeList& row = algopresccursor.currentRow();     
      //row.toOutputStream( std::cout ) << std::endl;  
      for(unsigned int bitidx=0;bitidx<128;++bitidx){
	std::string algopresc=algoprescBase+int2str(bitidx,3);
	algoprescale.push_back(row[algopresc].data<unsigned int>());
      }
    }
    delete QueryAlgoPresc;
    //
    //select prescale_factor_tt_000,prescale_factor_tt_001..._63 from cms_gt.gt_run_presc_tech_view where runr=:runnumber and prescale_index=0;
    //    
    coral::IQuery* QueryTechPresc=gtschemaHandle.newQuery();
    QueryTechPresc->addToTableList(runpresctechviewname);
    coral::AttributeList qTechPrescOutput;
    std::string techprescBase("PRESCALE_FACTOR_TT_");
    for(unsigned int bitidx=0;bitidx<lumi::N_TRGTECHBIT;++bitidx){
      std::string techpresc=techprescBase+this->int2str(bitidx,3);
      qTechPrescOutput.extend(techpresc,typeid(unsigned int));
    }
    for(unsigned int bitidx=0;bitidx<lumi::N_TRGTECHBIT;++bitidx){
      std::string techpresc=techprescBase+int2str(bitidx,3);
      QueryTechPresc->addToOutputList(techpresc);
    }
    coral::AttributeList TechPrescbindVariable;
    TechPrescbindVariable.extend("runnumber",typeid(int));
    TechPrescbindVariable.extend("prescaleindex",typeid(int));
    TechPrescbindVariable["runnumber"].data<int>()=runnumber;
    TechPrescbindVariable["prescaleindex"].data<int>()=0;
    QueryTechPresc->setCondition("runnr =:runnumber AND prescale_index =:prescaleindex",TechPrescbindVariable);
    QueryTechPresc->defineOutput(qTechPrescOutput);
    coral::ICursor& techpresccursor=QueryTechPresc->execute();
    while( techpresccursor.next() ){
      const coral::AttributeList& row = techpresccursor.currentRow();     
      //row.toOutputStream( std::cout ) << std::endl;
      for(unsigned int bitidx=0;bitidx<lumi::N_TRGTECHBIT;++bitidx){
	std::string techpresc=techprescBase+int2str(bitidx,3);
	techprescale.push_back(row[techpresc].data<unsigned int>());
      }
    }
    delete QueryTechPresc;
    transaction.commit();
    delete trgsession;
    //
    //reprocess Algo name result filling unallocated trigger bit with string "False"
    //
    for(size_t algoidx=0;algoidx<lumi::N_TRGALGOBIT;++algoidx){
      std::map<unsigned int,std::string>::iterator pos=triggernamemap.find(algoidx);
      if(pos!=triggernamemap.end()){
	algonames.push_back(pos->second);
      }else{
	algonames.push_back("False");
      }
    }
    //
    //reprocess Tech name result filling unallocated trigger bit with string "False"  
    //
    std::stringstream ss;
    for(size_t techidx=0;techidx<lumi::N_TRGTECHBIT;++techidx){
      std::map<unsigned int,std::string>::iterator pos=techtriggernamemap.find(techidx);
      ss<<techidx;
      technames.push_back(ss.str());
      ss.str(""); //clear the string buffer after usage
    }
    //
    //cross check result size
    //
    if(algonames.size()!=lumi::N_TRGALGOBIT || technames.size()!=lumi::N_TRGTECHBIT){
      throw lumi::Exception("wrong number of bits","retrieveData","TRGWBM2DB");
    }
    if(algoprescale.size()!=lumi::N_TRGALGOBIT || techprescale.size()!=lumi::N_TRGTECHBIT){
      throw lumi::Exception("wrong number of prescale","retrieveData","TRGWBM2DB");
    }
    if(deadtimeresult.size()!=algocount.size() || deadtimeresult.size()!=techcount.size()){
      throw lumi::Exception("inconsistent number of LS","retrieveData","TRGWBM2DB");
    }
    //    
    //write data into lumi db
    //
    coral::ISessionProxy* lumisession=svc->connect(m_dest,coral::Update);
    unsigned int totalcmsls=deadtimeresult.size();
    try{
      lumisession->transaction().start(false);
      coral::ISchema& schema=lumisession->nominalSchema();
      lumi::idDealer idg(schema);
      coral::ITable& trgtable=schema.tableHandle(LumiNames::trgTableName());
      coral::AttributeList trgData;
      trgData.extend<unsigned long long>("TRG_ID");
      trgData.extend<unsigned int>("RUNNUM");
      trgData.extend<unsigned int>("CMSLSNUM");
      trgData.extend<unsigned int>("BITNUM");
      trgData.extend<std::string>("BITNAME");
      trgData.extend<unsigned int>("TRGCOUNT");
      trgData.extend<unsigned long long>("DEADTIME");
      trgData.extend<unsigned int>("PRESCALE");
      coral::IBulkOperation* trgInserter=trgtable.dataEditor().bulkInsert(trgData,totalcmsls*192);
      //loop over lumi LS
      
      unsigned long long& trg_id=trgData["TRG_ID"].data<unsigned long long>();
      unsigned int& trgrunnum=trgData["RUNNUM"].data<unsigned int>();
      unsigned int& cmslsnum=trgData["CMSLSNUM"].data<unsigned int>();
      unsigned int& bitnum=trgData["BITNUM"].data<unsigned int>();
      std::string& bitname=trgData["BITNAME"].data<std::string>();
      unsigned int& count=trgData["TRGCOUNT"].data<unsigned int>();
      unsigned long long& deadtime=trgData["DEADTIME"].data<unsigned long long>();
      unsigned int& prescale=trgData["PRESCALE"].data<unsigned int>();

      
      TriggerDeadCountResult::const_iterator deadIt;
      TriggerDeadCountResult::const_iterator deadBeg=deadtimeresult.begin();
      TriggerDeadCountResult::const_iterator deadEnd=deadtimeresult.end();
      unsigned int trglscount=0;      
      for(deadIt=deadBeg;deadIt!=deadEnd;++deadIt,++trglscount ){
	unsigned int cmslscount=trglscount+1;
	BITCOUNT& algoinbits=algocount[trglscount];
	BITCOUNT& techinbits=techcount[trglscount];
	unsigned int trgbitcount=0;
	BITCOUNT::const_iterator algoBitIt;
	BITCOUNT::const_iterator algoBitBeg=algoinbits.begin();
	BITCOUNT::const_iterator algoBitEnd=algoinbits.end();
	for(algoBitIt=algoBitBeg;algoBitIt!=algoBitEnd;++algoBitIt,++trgbitcount){
	  trg_id = idg.generateNextIDForTable(LumiNames::trgTableName());
	  deadtime=*deadIt;
	  trgrunnum = runnumber;
	  cmslsnum = cmslscount;
	  bitnum=trgbitcount;
	  bitname=algonames[trgbitcount];
	  count=*algoBitIt;
	  prescale=algoprescale[trgbitcount];
	  trgInserter->processNextIteration();	
	}
	BITCOUNT::const_iterator techBitIt;
	BITCOUNT::const_iterator techBitBeg=techinbits.begin();
	BITCOUNT::const_iterator techBitEnd=techinbits.end();
	for(techBitIt=techBitBeg;techBitIt!=techBitEnd;++techBitIt,++trgbitcount){
	  trg_id = idg.generateNextIDForTable(LumiNames::trgTableName());
	  deadtime=*deadIt;
	  trgrunnum = runnumber;
	  cmslsnum = cmslscount;
	  bitnum=trgbitcount;
	  bitname=technames[trgbitcount-lumi::N_TRGALGOBIT];
	  count=*techBitIt;
	  prescale=techprescale[trgbitcount-lumi::N_TRGALGOBIT];
	  trgInserter->processNextIteration();	
	}
      }
      trgInserter->flush();
      delete trgInserter;
    }catch( const coral::Exception& er){
      lumisession->transaction().rollback();
      delete lumisession;
      delete svc;
      throw er;
    }
    //delete detailInserter;
    lumisession->transaction().commit();
    delete lumisession;
    delete svc;
  }
  const std::string TRGWBM2DB::dataType() const{
    return "TRG";
  }
  const std::string TRGWBM2DB::sourceType() const{
    return "DB";
  }
  //utilities
  std::string TRGWBM2DB::int2str(unsigned int t, unsigned int width){
    std::stringstream ss;
    ss.width(width);
    ss.fill('0');
    ss<<t;
    return ss.str();
  }
  unsigned int TRGWBM2DB::str2int(const std::string& s){
    std::istringstream myStream(s);
    unsigned int i;
    if(myStream>>i){
      return i;
    }else{
      throw lumi::Exception(std::string("str2int error"),"str2int","TRGWBM2DB");
    }
  }
  TRGWBM2DB::~TRGWBM2DB(){}
}//ns lumi
#include "RecoLuminosity/LumiProducer/interface/DataPipeFactory.h"
DEFINE_EDM_PLUGIN(lumi::DataPipeFactory,lumi::TRGWBM2DB,"TRGWBM2DB");
#endif

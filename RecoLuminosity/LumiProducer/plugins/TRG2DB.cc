#ifndef RecoLuminosity_LumiProducer_TRG2DB_h 
#define RecoLuminosity_LumiProducer_TRG2DB_h 
#include "RelationalAccess/IConnectionService.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/AttributeSpecification.h"
#include "CoralBase/Exception.h"
#include "RelationalAccess/AccessMode.h"
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
#include <iostream>
#include <sstream>
#include <map>
namespace lumi{
  class TRG2DB : public DataPipe{
  public:
    explicit TRG2DB(const std::string& dest);
    virtual void retrieveData( unsigned int runnumber);
    virtual const std::string dataType() const;
    virtual const std::string sourceType() const;
    virtual ~TRG2DB();
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
    typedef unsigned int DEADCOUNT;
    typedef std::vector<DEADCOUNT> TriggerDeadCountResult;
    typedef std::vector<unsigned int> BITCOUNT;
    typedef std::vector<BITCOUNT> TriggerCountResult_Algo;
    typedef std::vector<BITCOUNT> TriggerCountResult_Tech;
  };//cl TRG2DB
  //
  //implementation
  //
 TRG2DB::TRG2DB(const std::string& dest):DataPipe(dest){}
  void TRG2DB::retrieveData( unsigned int runnumber){
    std::string runnumberstr=int2str(runnumber,6);
    //query source GT database
    coral::ConnectionService* svc=new coral::ConnectionService;
    lumi::DBConfig dbconf(*svc);
    if(!m_authpath.empty()){
      dbconf.setAuthentication(m_authpath);
    }
    coral::ISessionProxy* session=svc->connect(m_source, coral::ReadOnly);
    coral::ITypeConverter& tpc=session->typeConverter();
    tpc.setCppTypeForSqlType("unsigned int","NUMBER(10)");
    
    coral::AttributeList bindVariableList;
    bindVariableList.extend("runnumber",typeid(unsigned int));
    bindVariableList["runnumber"].data<unsigned int>()=runnumber;
    std::string gtmonschema("CMS_GT_MON");
    std::string algoviewname("GT_MON_TRIG_ALGO_VIEW");
    std::string techviewname("GT_MON_TRIG_TECH_VIEW");
    std::string deadviewname("GT_MON_TRIG_DEAD_VIEW");
    std::string celltablename("GT_CELL_LUMISEG");
    
    std::string gtschema("CMS_GT");
    std::string runtechviewname("GT_RUN_TECH_VIEW");
    std::string runalgoviewname("GT_RUN_ALGO_VIEW");
    std::string runprescalgoviewname("GT_RUN_PRESC_ALGO_VIEW");
    std::string runpresctechviewname("GT_RUN_PRESC_TECH_VIEW");

    //data exchange format
    lumi::TRG2DB::BITCOUNT mybitcount_algo;
    mybitcount_algo.reserve(128);
    lumi::TRG2DB::BITCOUNT mybitcount_tech; 
    mybitcount_tech.reserve(64);
    lumi::TRG2DB::TriggerNameResult_Algo algonames;
    algonames.reserve(128);
    lumi::TRG2DB::TriggerNameResult_Tech technames;
    technames.reserve(64);
    lumi::TRG2DB::PrescaleResult_Algo algoprescale;
    algoprescale.reserve(128);
    lumi::TRG2DB::PrescaleResult_Tech techprescale;
    techprescale.reserve(64);
    lumi::TRG2DB::TriggerCountResult_Algo algocount;
    algocount.reserve(1024);
    lumi::TRG2DB::TriggerCountResult_Tech techcount;
    techcount.reserve(1024);
    lumi::TRG2DB::TriggerDeadCountResult deadtimeresult;
    deadtimeresult.reserve(400);
    coral::ITransaction& transaction=session->transaction();
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
       query tables in schema cms_gt_mon
    **/
    transaction.start(true);
    coral::ISchema& gtmonschemaHandle=session->schema(gtmonschema);    
    if(!gtmonschemaHandle.existsView(algoviewname)){
      throw lumi::Exception(std::string("non-existing view ")+algoviewname,"retrieveData","TRG2DB");
    }
    if(!gtmonschemaHandle.existsView(techviewname)){
      throw lumi::Exception(std::string("non-existing view ")+techviewname,"retrieveData","TRG2DB");
    }
    if(!gtmonschemaHandle.existsView(deadviewname)){
      throw lumi::Exception(std::string("non-existing view ")+deadviewname,"retrieveData","TRG2DB");
    }
    if(!gtmonschemaHandle.existsTable(celltablename)){
      throw lumi::Exception(std::string("non-existing table ")+celltablename,"retrieveData","TRG2DB");
    }
    //
    //select counts,lsnr,algobit from cms_gt_mon.gt_mon_trig_algo_view where runnr=:runnumber order by lsnr,algobit;
    //
    coral::IQuery* Queryalgoview=gtmonschemaHandle.newQuery();
    Queryalgoview->addToTableList(algoviewname);
    coral::AttributeList qalgoOutput;
    qalgoOutput.extend("counts",typeid(unsigned int));
    qalgoOutput.extend("lsnr",typeid(unsigned int));
    qalgoOutput.extend("algobit",typeid(unsigned int));
    Queryalgoview->addToOutputList("counts");
    Queryalgoview->addToOutputList("lsnr");
    Queryalgoview->addToOutputList("algobit");
    Queryalgoview->setCondition("RUNNR =:runnumber",bindVariableList);
    Queryalgoview->addToOrderList("lsnr");
    Queryalgoview->addToOrderList("algobit");
    Queryalgoview->defineOutput(qalgoOutput);
    coral::ICursor& c=Queryalgoview->execute();
    
    unsigned int s=0;
    while( c.next() ){
      const coral::AttributeList& row = c.currentRow();     
      //row.toOutputStream( std::cout ) << std::endl;
      //unsigned int lsnr=row["lsnr"].data<unsigned int>();
      unsigned int count=row["counts"].data<unsigned int>();
      unsigned int algobit=row["algobit"].data<unsigned int>();
      mybitcount_algo.push_back(count);
      if(algobit==127){
	algocount.push_back(mybitcount_algo);
	mybitcount_algo.clear();
      }
      ++s;
    }
    if(s==0){
      c.close();
      delete Queryalgoview;
      transaction.commit();
      throw lumi::Exception(std::string("requested run ")+runnumberstr+std::string(" doesn't exist for algocounts"),"retrieveData","TRG2DB");
    }
    if( mybitcount_algo.size()!=128){
      delete Queryalgoview;
      transaction.commit();
      throw lumi::Exception(std::string("total number of algo bits is not 128"),"retrieveData","TRG2DB");
    }
    delete Queryalgoview;
    //
    //select counts,lsnr,techbit from cms_gt_mon.gt_mon_trig_tech_view where runnr=:runnumber order by lsnr,techbit;
    //
    coral::IQuery* Querytechview=gtmonschemaHandle.newQuery();
    Querytechview->addToTableList(techviewname);
    coral::AttributeList qtechOutput;
    qtechOutput.extend("counts",typeid(unsigned int));
    qtechOutput.extend("lsnr",typeid(unsigned int));
    qtechOutput.extend("techbit",typeid(unsigned int));
    Querytechview->addToOutputList("counts");
    Querytechview->addToOutputList("lsnr");
    Querytechview->addToOutputList("techbit");
    Querytechview->setCondition("RUNNR =:runnumber",bindVariableList);
    Querytechview->addToOrderList("lsnr");
    Querytechview->addToOrderList("techbit");
    Querytechview->defineOutput(qtechOutput);
    coral::ICursor& techcursor=Querytechview->execute();
    
    s=0;
    while( techcursor.next() ){
      const coral::AttributeList& row = techcursor.currentRow();     
      //row.toOutputStream( std::cout ) << std::endl;
      //unsigned int lsnr=row["lsnr"].data<unsigned int>();
      unsigned int count=row["counts"].data<unsigned int>();
      unsigned int techbit=row["techbit"].data<unsigned int>();
      mybitcount_tech.push_back(count);
      if(techbit==63){
	techcount.push_back(mybitcount_tech);
	mybitcount_tech.clear();
      }
      ++s;
    }
    if(s==0){
      techcursor.close();
      delete Querytechview;
      transaction.commit();
      throw lumi::Exception(std::string("requested run ")+runnumberstr+std::string(" doesn't exist for tecgcounts"),"retrieveData","TRG2DB");
    }
    if( mybitcount_tech.size()!=64){
      delete Querytechview;
      transaction.commit();
      throw lumi::Exception(std::string("total number of tech bits is not 64"),"retrieveData","TRG2DB");
    }
    delete Querytechview;
    //
    //select counts,lsnr from cms_gt_mon.gt_mon_trig_dead_view where runnr=:runnumber and deadcounter=:countername order by lsnr;
    //
    coral::IQuery* Querydeadview=gtmonschemaHandle.newQuery();
    Querydeadview->addToTableList(deadviewname);
    coral::AttributeList qdeadOutput;
    qdeadOutput.extend("counts",typeid(unsigned int));
    qdeadOutput.extend("lsnr",typeid(unsigned int));
    Querydeadview->addToOutputList("counts");
    Querydeadview->addToOutputList("lsnr");
    coral::AttributeList bindVariablesDead;
    bindVariablesDead.extend("runnumber",typeid(int));
    bindVariablesDead.extend("countername",typeid(std::string));
    bindVariablesDead["runnumber"].data<int>()=runnumber;
    bindVariablesDead["countername"].data<std::string>()=std::string("Deadtime");
    Querydeadview->setCondition("RUNNR =:runnumber AND DEADCOUNTER =:countername",bindVariablesDead);
    Querydeadview->addToOrderList("lsnr");
    Querydeadview->defineOutput(qdeadOutput);
    coral::ICursor& deadcursor=Querydeadview->execute();
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
      std::cout<<"requested run "<<runnumber<<" doesn't exist for deadcount, do nothing"<<std::endl;
      deadcursor.close();
      delete Querydeadview;
      transaction.commit();
      return;
    }
    //transaction.commit();
    delete Querydeadview;
    /**
       Part II
       query tables in schema cms_gt
    **/
    coral::ISchema& gtschemaHandle=session->schema(gtschema);
    if(!gtschemaHandle.existsView(runtechviewname)){
      throw lumi::Exception(std::string("non-existing view ")+runtechviewname,"retrieveData","TRG2DB");
    }
    if(!gtschemaHandle.existsView(runalgoviewname)){
      throw lumi::Exception(std::string("non-existing view ")+runalgoviewname,"retrieveData","TRG2DB");
    }
    if(!gtschemaHandle.existsView(runprescalgoviewname)){
      throw lumi::Exception(std::string("non-existing view ")+runprescalgoviewname,"retrieveData","TRG2DB");
    }
    if(!gtschemaHandle.existsView(runpresctechviewname)){
      throw lumi::Exception(std::string("non-existing view ")+runpresctechviewname,"retrieveData","TRG2DB");
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
    for(unsigned int bitidx=0;bitidx<128;++bitidx){
      std::string algopresc=algoprescBase+int2str(bitidx,3);
      qAlgoPrescOutput.extend(algopresc,typeid(unsigned int));
    }
    for(unsigned int bitidx=0;bitidx<128;++bitidx){
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
    for(unsigned int bitidx=0;bitidx<64;++bitidx){
      std::string techpresc=techprescBase+this->int2str(bitidx,3);
      qTechPrescOutput.extend(techpresc,typeid(unsigned int));
    }
    for(unsigned int bitidx=0;bitidx<64;++bitidx){
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
      for(unsigned int bitidx=0;bitidx<64;++bitidx){
	std::string techpresc=techprescBase+int2str(bitidx,3);
	techprescale.push_back(row[techpresc].data<unsigned int>());
      }
    }
    delete QueryTechPresc;
    transaction.commit();
    //reprocess Algo name result filling unallocated trigger bit with string "False"
    for(size_t algoidx=0;algoidx<128;++algoidx){
      std::map<unsigned int,std::string>::iterator pos=triggernamemap.find(algoidx);
      if(pos!=triggernamemap.end()){
	algonames.push_back(pos->second);
      }else{
	algonames.push_back("False");
      }
    }
    //reprocess Tech name result filling unallocated trigger bit with string "False"  
    std::stringstream ss;
    for(size_t techidx=0;techidx<64;++techidx){
      std::map<unsigned int,std::string>::iterator pos=techtriggernamemap.find(techidx);
      ss<<techidx;
      technames.push_back(ss.str());
      ss.str(""); //clear the string buffer after usage
    }
    //    
    //write data into lumi db
    //
    try{
      unsigned int totalcmsls=deadtimeresult.size();
      session->transaction().start(false);
      coral::ISchema& schema=session->nominalSchema();
      lumi::idDealer idg(schema);
      coral::ITable& trgtable=schema.tableHandle(LumiNames::trgTableName());
      coral::AttributeList trgData;
      trgData.extend<unsigned long long>("TRG_ID");
      trgData.extend<unsigned int>("RUNNUM");
      trgData.extend<unsigned int>("CMSLUMINUM");
      trgData.extend<unsigned int>("BITNUM");
      trgData.extend<std::string>("BITNAME");
      trgData.extend<unsigned long long>("COUNT");
      trgData.extend<unsigned long long>("DEADTIME");
      trgData.extend<unsigned int>("PRESCALE");
      coral::IBulkOperation* trgInserter=trgtable.dataEditor().bulkInsert(trgData,totalcmsls*192);
      //loop over lumi LS
      
      unsigned long long& trg_id=trgData["TRG_ID"].data<unsigned long long>();
      unsigned int& trgrunnum=trgData["RUNNUM"].data<unsigned int>();
      unsigned int& cmsluminum=trgData["CMSLUMINUM"].data<unsigned int>();
      unsigned int& bitnum=trgData["BITNUM"].data<unsigned int>();
      std::string& bitname=trgData["BITNAME"].data<std::string>();
      unsigned long long& count=trgData["COUNT"].data<unsigned long long>();
      unsigned long long& deadtime=trgData["DEADTIME"].data<unsigned long long>();
      unsigned int& prescale=trgData["PRESCALE"].data<unsigned int>();


      TriggerCountResult_Algo::const_iterator algoIt;
      TriggerCountResult_Algo::const_iterator algoBeg=algocount.begin();
      TriggerCountResult_Algo::const_iterator algoEnd=algocount.end();
      TriggerCountResult_Tech::const_iterator techIt;
      TriggerCountResult_Tech::const_iterator techBeg=techcount.begin();
      TriggerCountResult_Tech::const_iterator techEnd=techcount.end();
      unsigned int trglscount=0;
      for(algoIt=algoBeg;algoIt!=algoEnd;++algoIt){
	unsigned int cmslscount=trglscount+1;
	BITCOUNT::const_iterator algoBitIt;
	BITCOUNT::const_iterator algoBitBeg=algoIt->begin();
	BITCOUNT::const_iterator algoBitEnd=algoIt->end();
	unsigned int j=0;
	for(algoBitIt=algoBitBeg;algoBitIt!=algoBitEnd;++algoBitIt){
	  trg_id = idg.generateNextIDForTable(LumiNames::trgTableName());
	  trgrunnum = runnumber;
	  cmsluminum = cmslscount;
	  bitnum=j;
	  bitname=algonames[j];
	  count=*algoBitIt;
	  prescale=algoprescale[j];
	  trgInserter->processNextIteration();	
	  ++j;
	}
	deadtime=deadtimeresult[trglscount];;
	++trglscount;
      }
      trgInserter->flush();
      delete trgInserter;
    }catch( const coral::Exception& er){
      std::cout<<"database problem "<<er.what()<<std::endl;
      session->transaction().rollback();
      delete session;
      delete svc;
      throw er;
    }
    //delete detailInserter;
    session->transaction().commit();
    delete session;
    delete svc;
  }
  const std::string TRG2DB::dataType() const{
    return "TRG";
  }
  const std::string TRG2DB::sourceType() const{
    return "DB";
  }
  //utilities
  std::string TRG2DB::int2str(unsigned int t, unsigned int width){
    std::stringstream ss;
    ss.width(width);
    ss.fill('0');
    ss<<t;
    return ss.str();
  }
  unsigned int TRG2DB::str2int(const std::string& s){
    std::istringstream myStream(s);
    unsigned int i;
    if(myStream>>i){
      return i;
    }else{
      throw lumi::Exception(std::string("str2int error"),"str2int","TRG2DB");
    }
  }
  TRG2DB::~TRG2DB(){}
}//ns lumi
#include "RecoLuminosity/LumiProducer/interface/DataPipeFactory.h"
DEFINE_EDM_PLUGIN(lumi::DataPipeFactory,lumi::TRG2DB,"TRG2DB");
#endif

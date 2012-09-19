#ifndef RecoLuminosity_LumiProducer_TRGScalers2DB_h 
#define RecoLuminosity_LumiProducer_TRGScalers2DB_h 
#include "CoralBase/AttributeList.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/AttributeSpecification.h"
#include "CoralBase/Blob.h"
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
#include "RecoLuminosity/LumiProducer/interface/RevisionDML.h"
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <map>
namespace lumi{
  class TRGScalers2DB : public DataPipe{
  public:
    const static unsigned int COMMITLSINTERVAL=150; //commit interval in LS, totalrow=nsl*192
    const static unsigned int COMMITLSTRGINTERVAL=550; //commit interval in LS of schema2
    explicit TRGScalers2DB(const std::string& dest);
    virtual unsigned long long retrieveData( unsigned int runnumber);
    virtual const std::string dataType() const;
    virtual const std::string sourceType() const;
    virtual ~TRGScalers2DB();													  

    //per run information
    typedef std::vector<std::string> TriggerNameResult_Algo;
    typedef std::vector<std::string> TriggerNameResult_Tech;
    //per lumisection information
    typedef unsigned long long DEADCOUNT;
    typedef float DEADFRAC;
    typedef std::vector<DEADCOUNT> TriggerDeadCountResult;
    typedef std::vector<DEADFRAC> TriggerDeadFracResult;
    typedef std::vector<unsigned int> BITCOUNT;
    typedef std::vector<BITCOUNT> TriggerCountResult_Algo;
    typedef std::vector<BITCOUNT> TriggerCountResult_Tech;
    typedef std::map< unsigned int, std::vector<unsigned int> > PrescaleResult_Algo;
    typedef std::map< unsigned int, std::vector<unsigned int> > PrescaleResult_Tech;

    std::string int2str(unsigned int t,unsigned int width);
    unsigned int str2int(const std::string& s);
    void writeTrgData(coral::ISessionProxy* session,
		      unsigned int runnumber,
		      const std::string& source,
		      TriggerDeadCountResult::iterator deadtimesBeg,
		      TriggerDeadCountResult::iterator deadtimesEnd,
		      TRGScalers2DB::TriggerDeadFracResult& deadfracs,
		      TriggerNameResult_Algo& algonames, 
		      TriggerNameResult_Tech& technames,
		      TriggerCountResult_Algo& algocounts,
		      TriggerCountResult_Tech& techcounts,
		      PrescaleResult_Algo& prescalealgo,
		      PrescaleResult_Tech& prescaletech,
		      unsigned int commitintv);
    unsigned long long writeTrgDataToSchema2(coral::ISessionProxy* session,
			       unsigned int runnumber,
			       const std::string& source,
			       TriggerDeadCountResult::iterator deadtimesBeg,
			       TriggerDeadCountResult::iterator deadtimesEnd,
			       TRGScalers2DB::TriggerDeadFracResult& deadfracs,       
			       TriggerNameResult_Algo& algonames,
			       TriggerNameResult_Tech& technames,
			       TriggerCountResult_Algo& algocounts,
			       TriggerCountResult_Tech& techcounts,
			       PrescaleResult_Algo& prescalealgo, 
			       PrescaleResult_Tech& prescaletech,
			       unsigned int commitintv);
  };//cl TRGScalers2DB
  //
  //implementation
  //
  //deadtime fraction query
  //
  //select fraction,lumi_section,count_bx from cms_gt_mon.v_scalers_tcs_deadtime where run_number=:runnum and scaler_name='DeadtimeBeamActive' 
  //
  //select count_bx,lumi_section,scaler_index from cms_gt_mon.v_scalers_fdl_algo where run_number=:runnum order by scaler_index;
  //
  //
  //select count_bx,lumi_section,scaler_index from cms_gt_mon.v_scalers_fdl_tech where run_number=:runnum order by scaler_index;
  //
  //select algo_index,alias from cms_gt.gt_run_algo_view where runnumber=:runnumber //order by algo_index;
  //
  //select techtrig_index,name from cms_gt.gt_run_tech_view where runnumber=:runnumber //order by techtrig_index;
  //
  //select distinct(prescale_index) from cms_gt_mon.lumi_sections where run_number=:runnumber;
  //
  //select lumi_section,prescale_index from cms_gt_mon.lumi_sections where run_number=:runnumber;
  //
  //select prescale_factor_algo_000,prescale_factor_algo_001..._127 from cms_gt.gt_run_presc_algo_view where runr=:runnumber and prescale_index=:prescale_index;
  // {prescale_index:[pres]}
  //
  //select prescale_factor_tt_000,prescale_factor_tt_001..._63 from cms_gt.gt_run_presc_tech_view where runr=:runnumber and prescale_index=0;
  //    
  TRGScalers2DB::TRGScalers2DB(const std::string& dest):DataPipe(dest){}
  unsigned long long TRGScalers2DB::retrieveData( unsigned int runnumber){
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
    std::string gtmonschema("CMS_GT_MON");
    std::string algoviewname("V_SCALERS_FDL_ALGO");
    std::string techviewname("V_SCALERS_FDL_TECH");
    std::string deadviewname("V_SCALERS_TCS_DEADTIME");
    std::string lstablename("LUMI_SECTIONS");

    std::string gtschema("CMS_GT");
    std::string runtechviewname("GT_RUN_TECH_VIEW");
    std::string runalgoviewname("GT_RUN_ALGO_VIEW");
    std::string runprescalgoviewname("GT_RUN_PRESC_ALGO_VIEW");
    std::string runpresctechviewname("GT_RUN_PRESC_TECH_VIEW");

    //data exchange format
    lumi::TRGScalers2DB::PrescaleResult_Algo algoprescale;
    lumi::TRGScalers2DB::PrescaleResult_Tech techprescale;
    lumi::TRGScalers2DB::BITCOUNT mybitcount_algo;
    mybitcount_algo.reserve(lumi::N_TRGALGOBIT);
    lumi::TRGScalers2DB::BITCOUNT mybitcount_tech; 
    mybitcount_tech.reserve(lumi::N_TRGTECHBIT);
    lumi::TRGScalers2DB::TriggerNameResult_Algo algonames;
    algonames.reserve(lumi::N_TRGALGOBIT);
    lumi::TRGScalers2DB::TriggerNameResult_Tech technames;
    technames.reserve(lumi::N_TRGTECHBIT);
    lumi::TRGScalers2DB::TriggerCountResult_Algo algocount;
    //algocount.reserve(400);
    lumi::TRGScalers2DB::TriggerCountResult_Tech techcount;
    //techcount.reserve(400);
    lumi::TRGScalers2DB::TriggerDeadFracResult deadfracresult;
    deadfracresult.reserve(500);
    lumi::TRGScalers2DB::TriggerDeadCountResult deadtimeresult;
    deadtimeresult.reserve(500);
    coral::ITransaction& transaction=trgsession->transaction();
    transaction.start(true);
    /**
       Part I
       query tables in schema cms_gt_mon
    **/
    coral::ISchema& gtmonschemaHandle=trgsession->schema(gtmonschema);    
    if(!gtmonschemaHandle.existsView(algoviewname)){
      throw lumi::Exception(std::string("non-existing view ")+algoviewname,"retrieveData","TRGScalers2DB");
    }
    if(!gtmonschemaHandle.existsView(techviewname)){
      throw lumi::Exception(std::string("non-existing view ")+techviewname,"retrieveData","TRGScalers2DB");
    }
    if(!gtmonschemaHandle.existsView(deadviewname)){
      throw lumi::Exception(std::string("non-existing view ")+deadviewname,"retrieveData","TRGScalers2DB");
    }
    
    //
    //select count_bx,lumi_section,scaler_index from cms_gt_mon.v_scalers_fdl_algo where run_number=:runnum order by lumi_section,scaler_index;
    //note: scaler_index  0-127
    //note: lumi_section count from 1 
    //
    coral::AttributeList bindVariableList;
    bindVariableList.extend("runnumber",typeid(unsigned int));
    bindVariableList["runnumber"].data<unsigned int>()=runnumber;

    coral::IQuery* Queryalgoview=gtmonschemaHandle.newQuery();
    Queryalgoview->addToTableList(algoviewname);
    coral::AttributeList qalgoOutput;
    qalgoOutput.extend("COUNT_BX",typeid(unsigned int));
    qalgoOutput.extend("LUMI_SECTION",typeid(unsigned int));
    qalgoOutput.extend("SCALER_INDEX",typeid(unsigned int));
    Queryalgoview->addToOutputList("COUNT_BX");
    Queryalgoview->addToOutputList("LUMI_SECTION");
    Queryalgoview->addToOutputList("SCALER_INDEX");
    Queryalgoview->setCondition("RUN_NUMBER=:runnumber",bindVariableList);
    Queryalgoview->addToOrderList("LUMI_SECTION");
    Queryalgoview->addToOrderList("SCALER_INDEX");
    Queryalgoview->defineOutput(qalgoOutput);
    coral::ICursor& c=Queryalgoview->execute();
    
    unsigned int s=0;
    while( c.next() ){
      const coral::AttributeList& row = c.currentRow();     
      //row.toOutputStream( std::cout ) << std::endl;
      unsigned int lsnr=row["LUMI_SECTION"].data<unsigned int>();
      unsigned int count=row["COUNT_BX"].data<unsigned int>();
      unsigned int algobit=row["SCALER_INDEX"].data<unsigned int>();
      mybitcount_algo.push_back(count);
      if(algobit==(lumi::N_TRGALGOBIT-1)){
	++s;
	while(s!=lsnr){
	  std::cout<<"ALGO COUNT alert: found hole in LS range "<<s<<std::endl;
	  std::cout<<"    fill all algocount 0 for LS "<<s<<std::endl;
	  std::vector<unsigned int> tmpzero(lumi::N_TRGALGOBIT,0);
	  algocount.push_back(tmpzero);
	  ++s;
	}
	algocount.push_back(mybitcount_algo);
	mybitcount_algo.clear();
      }
    }
    if(s==0){
      c.close();
      delete Queryalgoview;
      transaction.commit();
      throw lumi::Exception(std::string("requested run ")+runnumberstr+std::string(" doesn't exist for algocounts"),"retrieveData","TRGScalers2DB");
    }
    delete Queryalgoview;
    //
    //select count_bx,lumi_section,scaler_index from cms_gt_mon.v_scalers_fdl_tech where run_number=:runnum order by lumi_section,scaler_index;
    //
    //note: techobit 0-63
    //note: lsnr count from 1 
    //
    coral::IQuery* Querytechview=gtmonschemaHandle.newQuery();
    Querytechview->addToTableList(techviewname);
    coral::AttributeList qtechOutput;
    qtechOutput.extend("COUNT_BX",typeid(unsigned int));
    qtechOutput.extend("LUMI_SECTION",typeid(unsigned int));
    qtechOutput.extend("SCALER_INDEX",typeid(unsigned int));
    Querytechview->addToOutputList("COUNT_BX");
    Querytechview->addToOutputList("LUMI_SECTION");
    Querytechview->addToOutputList("SCALER_INDEX");
    Querytechview->setCondition("RUN_NUMBER=:runnumber",bindVariableList);
    Querytechview->addToOrderList("LUMI_SECTION");
    Querytechview->addToOrderList("SCALER_INDEX");
    Querytechview->defineOutput(qtechOutput);
    coral::ICursor& techcursor=Querytechview->execute();    
    s=0;
    while( techcursor.next() ){
      const coral::AttributeList& row = techcursor.currentRow();     
      //row.toOutputStream( std::cout #include <stdexcept>) << std::endl;
      unsigned int lsnr=row["LUMI_SECTION"].data<unsigned int>();
      unsigned int count=row["COUNT_BX"].data<unsigned int>();
      unsigned int techbit=row["SCALER_INDEX"].data<unsigned int>();
      mybitcount_tech.push_back(count);
      if(techbit==(lumi::N_TRGTECHBIT-1)){
	++s;
	while(s!=lsnr){
	  std::cout<<"TECH COUNT alert: found hole in LS range "<<s<<std::endl;
	  std::cout<<"     fill all techcount with 0 for LS "<<s<<std::endl;
	  std::vector<unsigned int> tmpzero(lumi::N_TRGTECHBIT,0);
	  techcount.push_back(tmpzero);
	  ++s;
	}
	techcount.push_back(mybitcount_tech);
	mybitcount_tech.clear();
      }
    }
    if(s==0){
      techcursor.close();
      delete Querytechview;
      transaction.commit();
      throw lumi::Exception(std::string("requested run ")+runnumberstr+std::string(" doesn't exist for techcounts"),"retrieveData","TRGScalers2DB");
    }
    delete Querytechview;

    //
    //select fraction,lumi_section,count_bx from cms_gt_mon.v_scalers_tcs_deadtime where run_number=:runnum and scaler_name='DeadtimeBeamActive' order by lumi_section' 
    //
    //note: lsnr count from 1 
    //
    coral::IQuery* Querydeadview=gtmonschemaHandle.newQuery();
    Querydeadview->addToTableList(deadviewname);
    coral::AttributeList qdeadOutput;
    qdeadOutput.extend("FRACTION",typeid(float));
    qdeadOutput.extend("LUMI_SECTION",typeid(unsigned int));
    qdeadOutput.extend("COUNT_BX",typeid(unsigned int));
    Querydeadview->addToOutputList("FRACTION");
    Querydeadview->addToOutputList("LUMI_SECTION");
    Querydeadview->addToOutputList("COUNT_BX");
    coral::AttributeList bindVariablesDead;
    bindVariablesDead.extend("runnumber",typeid(int));
    bindVariablesDead.extend("scalername",typeid(std::string));
    bindVariablesDead["runnumber"].data<int>()=runnumber;
    bindVariablesDead["scalername"].data<std::string>()=std::string("DeadtimeBeamActive");
    Querydeadview->setCondition("RUN_NUMBER=:runnumber AND SCALER_NAME=:scalername",bindVariablesDead);
    Querydeadview->addToOrderList("LUMI_SECTION");
    Querydeadview->defineOutput(qdeadOutput);
    coral::ICursor& deadcursor=Querydeadview->execute();
    s=0;
    while( deadcursor.next() ){
      const coral::AttributeList& row = deadcursor.currentRow();     
      //row.toOutputStream( std::cout ) << std::endl;
      ++s;
      unsigned int lsnr=row["LUMI_SECTION"].data<unsigned int>();
      float dfrac=1.0;
      unsigned int count=0;
      while(s!=lsnr){
	std::cout<<"DEADTIME alert: found hole in LS range"<<s<<std::endl;        
	std::cout<<"         fill deadtimefraction 0%, deadtimebeamactive 0 for LS "<<s<<std::endl;
	deadfracresult.push_back(dfrac);
	deadtimeresult.push_back(count);
	++s;
      }
      if(!row["FRACTION"].isNull()){
	dfrac=row["FRACTION"].data<float>(); //deadfraction is null from trigger means "undefined", but insert 1.0...
      }else{
	std::cout<<"DEADTIME fraction alert: undefined fraction , assume 100% , LS "<<lsnr<<std::endl;
      }
      if(dfrac>1.0){
	std::cout<<"DEADTIME fraction alert: overflow dead fraction , force to 100% , LS "<<lsnr<<std::endl;
	dfrac=1.0;
      }
      deadfracresult.push_back(dfrac);
      count=row["COUNT_BX"].data<unsigned int>();
      deadtimeresult.push_back(count);
    }
    if(s==0){
      deadcursor.close();
      delete Querydeadview;
      transaction.commit();
      throw lumi::Exception(std::string("requested run ")+runnumberstr+std::string(" doesn't exist for deadcounts"),"retrieveData","TRGScalers2DB");
      return 0;
    }
    delete Querydeadview;
    
    //
    //select distinct(prescale_index) from cms_gt_mon.lumi_sections where run_number=:runnumber;
    //
    std::vector< int > prescidx;
    coral::IQuery* allpsidxQuery=gtmonschemaHandle.newQuery();
    allpsidxQuery->addToTableList(lstablename);
    coral::AttributeList allpsidxOutput;
    allpsidxOutput.extend("psidx",typeid(int));
    allpsidxQuery->addToOutputList("distinct(PRESCALE_INDEX)","psidx");
    coral::AttributeList bindVariablesAllpsidx;
    bindVariablesAllpsidx.extend("runnumber",typeid(int));
    bindVariablesAllpsidx["runnumber"].data<int>()=runnumber;
    allpsidxQuery->setCondition("RUN_NUMBER =:runnumber",bindVariablesAllpsidx);
    allpsidxQuery->defineOutput(allpsidxOutput);
    coral::ICursor& allpsidxCursor=allpsidxQuery->execute();
    while( allpsidxCursor.next() ){
      const coral::AttributeList& row = allpsidxCursor.currentRow();     
      int psidx=row["psidx"].data<int>();
      prescidx.push_back(psidx);
    }
    delete allpsidxQuery;
    std::map< int, std::vector<unsigned int> > algoprescMap;
    std::map< int, std::vector<unsigned int> > techprescMap;
    std::vector< int >::iterator prescidxIt;
    std::vector< int >::iterator prescidxItBeg=prescidx.begin();
    std::vector< int >::iterator prescidxItEnd=prescidx.end();
    for(prescidxIt=prescidxItBeg;prescidxIt!=prescidxItEnd;++prescidxIt){
      std::vector<unsigned int> algopres; algopres.reserve(lumi::N_TRGALGOBIT);
      std::vector<unsigned int> techpres; techpres.reserve(lumi::N_TRGTECHBIT);
      algoprescMap.insert(std::make_pair(*prescidxIt,algopres));
      techprescMap.insert(std::make_pair(*prescidxIt,techpres));
    }
    //
    //select lumi_section,prescale_index from cms_gt_mon.lumi_sections where run_number=:runnumber
    // {ls:prescale_index}
    //
    std::map< unsigned int, int > lsprescmap;
    coral::IQuery* lstoprescQuery=gtmonschemaHandle.newQuery();
    lstoprescQuery->addToTableList(lstablename);
    coral::AttributeList lstoprescOutput;
    lstoprescOutput.extend("lumisection",typeid(unsigned int));
    lstoprescOutput.extend("psidx",typeid(int));
    lstoprescQuery->addToOutputList("LUMI_SECTION","lumisection");
    lstoprescQuery->addToOutputList("PRESCALE_INDEX","psidx");
    coral::AttributeList bindVariablesLstopresc;
    bindVariablesLstopresc.extend("runnumber",typeid(int));
    bindVariablesLstopresc["runnumber"].data<int>()=runnumber;
    lstoprescQuery->setCondition("RUN_NUMBER =:runnumber",bindVariablesLstopresc);
    lstoprescQuery->defineOutput(lstoprescOutput);
    unsigned int lsprescount=0;
    unsigned int lastpresc=0;
    coral::ICursor& lstoprescCursor=lstoprescQuery->execute();
    while( lstoprescCursor.next() ){
      ++lsprescount;
      const coral::AttributeList& row = lstoprescCursor.currentRow();     
      unsigned int lumisection=row["lumisection"].data< unsigned int>();
      while(lsprescount!=lumisection){
	  std::cout<<"PRESCALE_INDEX COUNT alert: found hole in LS range "<<lsprescount<<std::endl;
	  std::cout<<"     fill this prescale from last availabl prescale "<<lastpresc<<std::endl;
	  unsigned int guesspsidx=lastpresc;
	  lsprescmap.insert(std::make_pair(lsprescount,guesspsidx));
	  ++lsprescount;
      }
      int psidx=row["psidx"].data< int>();
      lsprescmap.insert(std::make_pair(lumisection,psidx));
      lastpresc=psidx;
    }
    if(lsprescount==0){
      lstoprescCursor.close();
      delete lstoprescQuery ;
      transaction.commit();
      throw lumi::Exception(std::string("requested run ")+runnumberstr+std::string(" doesn't exist for prescale_index"),"retrieveData","TRG2DB");
    }
    delete lstoprescQuery ;

    for(prescidxIt=prescidxItBeg;prescidxIt!=prescidxItEnd;++prescidxIt){
      std::vector<unsigned int> algopres; algopres.reserve(lumi::N_TRGALGOBIT);
      std::vector<unsigned int> techpres; techpres.reserve(lumi::N_TRGTECHBIT);
      algoprescMap.insert(std::make_pair(*prescidxIt,algopres));
      techprescMap.insert(std::make_pair(*prescidxIt,techpres));
    }
    //prefill lsprescmap
    //transaction.commit();
    /**
       Part II
       query tables in schema cms_gt
    **/
    coral::ISchema& gtschemaHandle=trgsession->schema(gtschema);
    if(!gtschemaHandle.existsView(runtechviewname)){
      throw lumi::Exception(std::string("non-existing view ")+runtechviewname,"str2int","TRG2DB");
    }
    if(!gtschemaHandle.existsView(runalgoviewname)){
      throw lumi::Exception(std::string("non-existing view ")+runalgoviewname,"str2int","TRG2DB");
    }
    if(!gtschemaHandle.existsView(runprescalgoviewname)){
      throw lumi::Exception(std::string("non-existing view ")+runprescalgoviewname,"str2int","TRG2DB");
    }
    if(!gtschemaHandle.existsView(runpresctechviewname)){
      throw lumi::Exception(std::string("non-existing view ")+runpresctechviewname,"str2int","TRG2DB");
    }
    //
    //select algo_index,alias from cms_gt.gt_run_algo_view where runnumber=:runnumber //order by algo_index;
    //
    std::map<unsigned int,std::string> triggernamemap;
    coral::IQuery* QueryName=gtschemaHandle.newQuery();
    QueryName->addToTableList(runalgoviewname);
    coral::AttributeList qAlgoNameOutput;
    qAlgoNameOutput.extend("ALGO_INDEX",typeid(unsigned int));
    qAlgoNameOutput.extend("ALIAS",typeid(std::string));
    QueryName->addToOutputList("ALGO_INDEX");
    QueryName->addToOutputList("ALIAS");
    QueryName->setCondition("RUNNUMBER=:runnumber",bindVariableList);
    //QueryName->addToOrderList("algo_index");
    QueryName->defineOutput(qAlgoNameOutput);
    coral::ICursor& algonamecursor=QueryName->execute();
    while( algonamecursor.next() ){
      const coral::AttributeList& row = algonamecursor.currentRow();     
      //row.toOutputStream( std::cout ) << std::endl;
      unsigned int algo_index=row["ALGO_INDEX"].data<unsigned int>();
      std::string algo_name=row["ALIAS"].data<std::string>();
      triggernamemap.insert(std::make_pair(algo_index,algo_name));
    }
    delete QueryName;
    
    //
    //select techtrig_index,name from cms_gt.gt_run_tech_view where runnumber=:runnumber //order by techtrig_index;
    //
    std::map<unsigned int,std::string> techtriggernamemap;
    coral::IQuery* QueryTechName=gtschemaHandle.newQuery();
    QueryTechName->addToTableList(runtechviewname);
    coral::AttributeList qTechNameOutput;
    qTechNameOutput.extend("TECHTRIG_INDEX",typeid(unsigned int));
    qTechNameOutput.extend("NAME",typeid(std::string));
    QueryTechName->addToOutputList("TECHTRIG_INDEX");
    QueryTechName->addToOutputList("NAME");
    QueryTechName->setCondition("RUNNUMBER=:runnumber",bindVariableList);
    //QueryTechName->addToOrderList("techtrig_index");
    QueryTechName->defineOutput(qTechNameOutput);
    coral::ICursor& technamecursor=QueryTechName->execute();
    while( technamecursor.next() ){
      const coral::AttributeList& row = technamecursor.currentRow();     
      //row.toOutputStream( std::cout ) << std::endl;
      unsigned int tech_index=row["TECHTRIG_INDEX"].data<unsigned int>();
      std::string tech_name=row["NAME"].data<std::string>();
      techtriggernamemap.insert(std::make_pair(tech_index,tech_name));
    }
    delete QueryTechName;
    //
    //loop over all prescale_index
    //
    //select prescale_factor_algo_000,prescale_factor_algo_001..._127 from cms_gt.gt_run_presc_algo_view where runr=:runnumber and prescale_index=:prescale_index;
    // {prescale_index:[pres]}
    //
    std::vector< int >::iterator presIt;
    std::vector< int >::iterator presItBeg=prescidx.begin();
    std::vector< int >::iterator presItEnd=prescidx.end();
    for( presIt=presItBeg; presIt!=presItEnd; ++presIt ){
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
      PrescbindVariable["prescaleindex"].data<int>()=*presIt;
      QueryAlgoPresc->setCondition("RUNNR=:runnumber AND PRESCALE_INDEX=:prescaleindex",PrescbindVariable);
      QueryAlgoPresc->defineOutput(qAlgoPrescOutput);
      coral::ICursor& algopresccursor=QueryAlgoPresc->execute();
      while( algopresccursor.next() ){
	const coral::AttributeList& row = algopresccursor.currentRow();     
	for(unsigned int bitidx=0;bitidx<lumi::N_TRGALGOBIT;++bitidx){
	  std::string algopresc=algoprescBase+int2str(bitidx,3);
	  algoprescMap[*presIt].push_back(row[algopresc].data<unsigned int>());
	}
      }
      delete QueryAlgoPresc;
    }
    //
    //select prescale_factor_tt_000,prescale_factor_tt_001..._63 from cms_gt.gt_run_presc_tech_view where runr=:runnumber and prescale_index=0;
    //    
    for( presIt=presItBeg; presIt!=presItEnd; ++presIt ){
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
      TechPrescbindVariable["prescaleindex"].data<int>()=*presIt;
      QueryTechPresc->setCondition("RUNNR=:runnumber AND PRESCALE_INDEX=:prescaleindex",TechPrescbindVariable);
      QueryTechPresc->defineOutput(qTechPrescOutput);
      coral::ICursor& techpresccursor=QueryTechPresc->execute();
      while( techpresccursor.next() ){
	const coral::AttributeList& row = techpresccursor.currentRow();     
	//row.toOutputStream( std::cout ) << std::endl;
	for(unsigned int bitidx=0;bitidx<lumi::N_TRGTECHBIT;++bitidx){
	  std::string techpresc=techprescBase+int2str(bitidx,3);
	  techprescMap[*presIt].push_back(row[techpresc].data<unsigned int>());
	}
      }
      delete QueryTechPresc;
    }
    transaction.commit();
    delete trgsession;

    std::map< unsigned int, int >::iterator lsprescmapIt;
    std::map< unsigned int, int >::iterator lsprescmapItBeg=lsprescmap.begin();
    std::map< unsigned int, int >::iterator lsprescmapItEnd=lsprescmap.end();
    for( lsprescmapIt=lsprescmapItBeg; lsprescmapIt!=lsprescmapItEnd; ++lsprescmapIt ){
      unsigned int ls=lsprescmapIt->first;
      int preidx=lsprescmapIt->second;
      algoprescale.insert(std::make_pair(ls,algoprescMap[preidx]));
      techprescale.insert(std::make_pair(ls,techprescMap[preidx]));
    }
    algoprescMap.clear();
    techprescMap.clear();
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
      ss<<techidx;
      technames.push_back(ss.str());
      ss.str(""); //clear the string buffer after usage
    }
    //
    //cross check result size
    //
    if(algonames.size()!=lumi::N_TRGALGOBIT || technames.size()!=lumi::N_TRGTECHBIT){
      throw lumi::Exception("wrong number of bits","retrieveData","TRG2DB");
    }
    //if(algoprescale.size()!=lumi::N_TRGALGOBIT || techprescale.size()!=lumi::N_TRGTECHBIT){
    //  throw lumi::Exception("wrong number of prescale","retrieveData","TRG2DB");
    //}
    if(deadtimeresult.size()!=deadfracresult.size()|| deadtimeresult.size()!=algocount.size() || deadtimeresult.size()!=techcount.size() || deadtimeresult.size()!=algoprescale.size() || deadtimeresult.size()!=techprescale.size() ){
      //throw lumi::Exception("inconsistent number of LS","retrieveData","TRG2DB");
      std::cout<<"[WARNING] inconsistent number of LS of deadtimecounter,deadfrac,algo,tech,prescalealgo,prescaletech "<<deadtimeresult.size()<<" "<<deadfracresult.size()<<" "<<algocount.size()<<" "<<techcount.size()<<" "<<algoprescale.size()<<" "<<techprescale.size()<<std::endl;
      TRGScalers2DB::TriggerDeadCountResult::iterator dIt;
      TRGScalers2DB::TriggerDeadCountResult::iterator dBeg=deadtimeresult.begin();
      TRGScalers2DB::TriggerDeadCountResult::iterator dEnd=deadtimeresult.end();
      unsigned int dcnt=0;
      for(dIt=dBeg;dIt!=dEnd;++dIt){
	try{
	  deadfracresult.at(dcnt);
	}catch(std::out_of_range& er){
	  std::cout<<"[WARNING] filling FAKE deadfrac=0.0 at LS "<<dcnt<<std::endl;
	  deadfracresult[dcnt]=0.0;
	}
	try{
	  algocount.at(dcnt);
	}catch(std::out_of_range& er){
	  std::vector<unsigned int> tmpzero(lumi::N_TRGALGOBIT,0);
	  std::cout<<"[WARNING] filling FAKE algocount at LS "<<dcnt<<std::endl;
	  algocount[dcnt]=tmpzero;
	}
	try{
	  techcount.at(dcnt);
	}catch(std::out_of_range& er){
	  std::vector<unsigned int> tmpzero(lumi::N_TRGTECHBIT,0);
	  std::cout<<"[WARNING] filling FAKE techcount at LS "<<dcnt<<std::endl;
	  techcount[dcnt]=tmpzero;
	}      
	if(algoprescale.find(dcnt+1)==algoprescale.end()){
	  std::vector<unsigned int> tmpzero(lumi::N_TRGALGOBIT,1);
	  std::cout<<"[WARNING] filling FAKE 1 algoprescale at LS "<<dcnt+1<<std::endl;
	  algoprescale[dcnt+1]=tmpzero;
	}
	if(techprescale.find(dcnt+1)==techprescale.end()){
	  std::vector<unsigned int> tmpzero(lumi::N_TRGTECHBIT,1);
	  std::cout<<"[WARNING] filling FAKE 1 techprescale at LS "<<dcnt+1<<std::endl;
	  techprescale[dcnt+1]=tmpzero;
	}
	++dcnt;	
      }
    }
 //    
    //write data into lumi db
    //
    unsigned long long trgdataid=0;
    coral::ISessionProxy* lumisession=svc->connect(m_dest,coral::Update);
    coral::ITypeConverter& lumitpc=lumisession->typeConverter();
    lumitpc.setCppTypeForSqlType("unsigned int","NUMBER(7)");
    lumitpc.setCppTypeForSqlType("unsigned int","NUMBER(10)");
    lumitpc.setCppTypeForSqlType("unsigned long long","NUMBER(20)");
    try{
      if(m_mode=="loadoldschema"){ 
	 std::cout<<"writing trg data to old trg table "<<std::endl;
	 writeTrgData(lumisession,runnumber,m_source,deadtimeresult.begin(),deadtimeresult.end(),deadfracresult,algonames,technames,algocount,techcount,algoprescale,techprescale,COMMITLSINTERVAL);
	 std::cout<<"done"<<std::endl;
      }
      std::cout<<"writing trg data to new lstrg table "<<std::endl;
      trgdataid=writeTrgDataToSchema2(lumisession,runnumber,m_source,deadtimeresult.begin(),deadtimeresult.end(),deadfracresult,algonames,technames,algocount,techcount,algoprescale,techprescale,COMMITLSTRGINTERVAL);
      std::cout<<"done"<<std::endl;
      delete lumisession;
      delete svc;
    }catch( const coral::Exception& er){
      std::cout<<"database error "<<er.what()<<std::endl;
      lumisession->transaction().rollback();
      delete lumisession;
      delete svc;
      throw er;
    }
    return trgdataid;
  }
  void  
  TRGScalers2DB::writeTrgData(coral::ISessionProxy* lumisession,
		       unsigned int runnumber,
		       const std::string& source,
		       TRGScalers2DB::TriggerDeadCountResult::iterator deadtimesBeg,TRGScalers2DB::TriggerDeadCountResult::iterator deadtimesEnd,
		       TRGScalers2DB::TriggerDeadFracResult& deadfracs,
		       TRGScalers2DB::TriggerNameResult_Algo& algonames,
		       TRGScalers2DB::TriggerNameResult_Tech& technames,
		       TRGScalers2DB::TriggerCountResult_Algo& algocounts,
		       TRGScalers2DB::TriggerCountResult_Tech& techcounts,
		       TRGScalers2DB::PrescaleResult_Algo& prescalealgo,
		       TRGScalers2DB::PrescaleResult_Tech& prescaletech,
		       unsigned int commitintv){    
    TRGScalers2DB::TriggerDeadCountResult::iterator deadIt;
    //unsigned int totalcmsls=deadtimes.size();
    unsigned int totalcmsls=std::distance(deadtimesBeg,deadtimesEnd);
    std::cout<<"inserting totalcmsls "<<totalcmsls<<std::endl;
    std::map< unsigned long long, std::vector<unsigned long long> > idallocationtable;
    std::cout<<"\t allocating total ids "<<totalcmsls*lumi::N_TRGBIT<<std::endl; 
    lumisession->transaction().start(false);
    lumi::idDealer idg(lumisession->nominalSchema());
    unsigned long long trgID = idg.generateNextIDForTable(LumiNames::trgTableName(),totalcmsls*lumi::N_TRGBIT)-totalcmsls*lumi::N_TRGBIT;
    //lumisession->transaction().commit();
    unsigned int trglscount=0;
    for(deadIt=deadtimesBeg;deadIt!=deadtimesEnd;++deadIt,++trglscount){
      std::vector<unsigned long long> bitvec;
      bitvec.reserve(lumi::N_TRGBIT);
      const BITCOUNT& algoinbits=algocounts[trglscount];
      const BITCOUNT& techinbits=techcounts[trglscount];
      BITCOUNT::const_iterator algoBitIt;
      BITCOUNT::const_iterator algoBitBeg=algoinbits.begin();
      BITCOUNT::const_iterator algoBitEnd=algoinbits.end();	
      for(algoBitIt=algoBitBeg;algoBitIt!=algoBitEnd;++algoBitIt,++trgID){
	bitvec.push_back(trgID);
      }
      BITCOUNT::const_iterator techBitIt;
      BITCOUNT::const_iterator techBitBeg=techinbits.begin();
      BITCOUNT::const_iterator techBitEnd=techinbits.end();
      for(techBitIt=techBitBeg;techBitIt!=techBitEnd;++techBitIt,++trgID){
	bitvec.push_back(trgID);
      }
      idallocationtable.insert(std::make_pair(trglscount,bitvec));
    }
    std::cout<<"\t all ids allocated"<<std::endl; 
    coral::AttributeList trgData;
    trgData.extend<unsigned long long>("TRG_ID");
    trgData.extend<unsigned int>("RUNNUM");
    trgData.extend<unsigned int>("CMSLSNUM");
    trgData.extend<unsigned int>("BITNUM");
    trgData.extend<std::string>("BITNAME");
    trgData.extend<unsigned int>("TRGCOUNT");
    trgData.extend<unsigned long long>("DEADTIME");
    trgData.extend<float>("DEADFRAC");
    trgData.extend<unsigned int>("PRESCALE");
    
    unsigned long long& trg_id=trgData["TRG_ID"].data<unsigned long long>();
    unsigned int& trgrunnum=trgData["RUNNUM"].data<unsigned int>();
    unsigned int& cmslsnum=trgData["CMSLSNUM"].data<unsigned int>();
    unsigned int& bitnum=trgData["BITNUM"].data<unsigned int>();
    std::string& bitname=trgData["BITNAME"].data<std::string>();
    unsigned int& count=trgData["TRGCOUNT"].data<unsigned int>();
    unsigned long long& deadtime=trgData["DEADTIME"].data<unsigned long long>();
    float& deadfrac=trgData["DEADFRAC"].data<float>();
    unsigned int& prescale=trgData["PRESCALE"].data<unsigned int>();
                      
    trglscount=0;    
    coral::IBulkOperation* trgInserter=0; 
    unsigned int comittedls=0;
    for(deadIt=deadtimesBeg;deadIt!=deadtimesEnd;++deadIt,++trglscount ){
      unsigned int cmslscount=trglscount+1;
      float dfra=deadfracs[trglscount];
      const BITCOUNT& algoinbits=algocounts[trglscount];
      const BITCOUNT& techinbits=techcounts[trglscount];
      unsigned int trgbitcount=0;
      BITCOUNT::const_iterator algoBitIt;
      BITCOUNT::const_iterator algoBitBeg=algoinbits.begin();
      BITCOUNT::const_iterator algoBitEnd=algoinbits.end();
      if(!lumisession->transaction().isActive()){ 
	lumisession->transaction().start(false);
	coral::ITable& trgtable=lumisession->nominalSchema().tableHandle(LumiNames::trgTableName());
	trgInserter=trgtable.dataEditor().bulkInsert(trgData,2048);
      }else{
	if(deadIt==deadtimesBeg){
	  coral::ITable& trgtable=lumisession->nominalSchema().tableHandle(LumiNames::trgTableName());
	  trgInserter=trgtable.dataEditor().bulkInsert(trgData,2048);
	}
      }
      for(algoBitIt=algoBitBeg;algoBitIt!=algoBitEnd;++algoBitIt,++trgbitcount){
	trg_id = idallocationtable[trglscount].at(trgbitcount);
	deadtime=*deadIt;
	deadfrac=dfra;
	trgrunnum = runnumber;
	cmslsnum = cmslscount;
	bitnum=trgbitcount;
	bitname=algonames[trgbitcount];
	count=*algoBitIt;
	prescale=prescalealgo[cmslscount].at(trgbitcount);
	//std::cout<<"cmslsnum "<<cmslsnum<<" bitnum "<<bitnum<<" bitname "<<bitname<<" prescale "<< prescale<<" count "<<count<<std::endl;
	trgInserter->processNextIteration();	
      }
      BITCOUNT::const_iterator techBitIt;
      BITCOUNT::const_iterator techBitBeg=techinbits.begin();
      BITCOUNT::const_iterator techBitEnd=techinbits.end();
      for(techBitIt=techBitBeg;techBitIt!=techBitEnd;++techBitIt,++trgbitcount){
	trg_id = idallocationtable[trglscount].at(trgbitcount);
	deadtime=*deadIt;
	deadfrac=dfra;
	trgrunnum = runnumber;
	cmslsnum = cmslscount;
	bitnum=trgbitcount;
	bitname=technames[trgbitcount-lumi::N_TRGALGOBIT];
	count=*techBitIt;
	prescale=prescaletech[cmslsnum][trgbitcount-lumi::N_TRGALGOBIT];
	trgInserter->processNextIteration();	
      }
      trgInserter->flush();
      ++comittedls;
      if(comittedls==commitintv){
	std::cout<<"\t committing in LS chunck "<<comittedls<<std::endl; 
	delete trgInserter; trgInserter=0;
	lumisession->transaction().commit();
	comittedls=0;
	std::cout<<"\t committed "<<std::endl; 
      }else if( trglscount==( totalcmsls-1) ){
	std::cout<<"\t committing at the end"<<std::endl; 
	delete trgInserter; trgInserter=0;
	lumisession->transaction().commit();
	std::cout<<"\t done"<<std::endl; 
      }
    }
  }
  unsigned long long
  TRGScalers2DB::writeTrgDataToSchema2(coral::ISessionProxy* lumisession,
				unsigned int irunnumber,
				const std::string& source,
			        TriggerDeadCountResult::iterator deadtimesBeg,
				TriggerDeadCountResult::iterator deadtimesEnd,
				TRGScalers2DB::TriggerDeadFracResult& deadfracs,       
				TRGScalers2DB::TriggerNameResult_Algo& algonames, 
				TRGScalers2DB::TriggerNameResult_Tech& technames,
				TRGScalers2DB::TriggerCountResult_Algo& algocounts,
				TRGScalers2DB::TriggerCountResult_Tech& techcounts,
				TRGScalers2DB::PrescaleResult_Algo& prescalealgo,
				TRGScalers2DB::PrescaleResult_Tech& prescaletech,
				unsigned int commitintv){
    TRGScalers2DB::TriggerDeadCountResult::iterator deadIt;
    unsigned int totalcmsls=std::distance(deadtimesBeg,deadtimesEnd);
    std::cout<<"inserting totalcmsls "<<totalcmsls<<std::endl;
    coral::AttributeList lstrgData;
    lstrgData.extend<unsigned long long>("DATA_ID");
    lstrgData.extend<unsigned int>("RUNNUM");
    lstrgData.extend<unsigned int>("CMSLSNUM");
    lstrgData.extend<unsigned long long>("DEADTIMECOUNT");
    lstrgData.extend<unsigned int>("BITZEROCOUNT");
    lstrgData.extend<unsigned int>("BITZEROPRESCALE");
    lstrgData.extend<float>("DEADFRAC");
    lstrgData.extend<coral::Blob>("PRESCALEBLOB");
    lstrgData.extend<coral::Blob>("TRGCOUNTBLOB");

    unsigned long long& data_id=lstrgData["DATA_ID"].data<unsigned long long>();
    unsigned int& trgrunnum=lstrgData["RUNNUM"].data<unsigned int>();
    unsigned int& cmslsnum=lstrgData["CMSLSNUM"].data<unsigned int>();
    unsigned long long& deadtime=lstrgData["DEADTIMECOUNT"].data<unsigned long long>();
    unsigned int& bitzerocount=lstrgData["BITZEROCOUNT"].data<unsigned int>();
    unsigned int& bitzeroprescale=lstrgData["BITZEROPRESCALE"].data<unsigned int>();
    float& deadfrac=lstrgData["DEADFRAC"].data<float>();
    coral::Blob& prescaleblob=lstrgData["PRESCALEBLOB"].data<coral::Blob>();
    coral::Blob& trgcountblob=lstrgData["TRGCOUNTBLOB"].data<coral::Blob>();

    unsigned long long branch_id=3;
    std::string branch_name("DATA");
    lumi::RevisionDML revisionDML;
    lumi::RevisionDML::TrgEntry trgrundata;
    std::stringstream op;
    op<<irunnumber;
    std::string runnumberStr=op.str();
    lumisession->transaction().start(false);
    trgrundata.entry_name=runnumberStr;
    trgrundata.source=source;
    trgrundata.runnumber=irunnumber;
    std::string bitnames;
    TriggerNameResult_Algo::iterator bitnameIt;
    TriggerNameResult_Algo::iterator bitnameItBeg=algonames.begin();
    TriggerNameResult_Algo::iterator bitnameItEnd=algonames.end();
    for (bitnameIt=bitnameItBeg;bitnameIt!=bitnameItEnd;++bitnameIt){
      if(bitnameIt!=bitnameItBeg){
	bitnames+=std::string(",");
      }
      bitnames+=*bitnameIt;
    }
    TriggerNameResult_Tech::iterator techbitnameIt;
    TriggerNameResult_Tech::iterator techbitnameItBeg=technames.begin();
    TriggerNameResult_Tech::iterator techbitnameItEnd=technames.end();
    for(techbitnameIt=techbitnameItBeg;techbitnameIt!=techbitnameItEnd;++techbitnameIt){
      bitnames+=std::string(",");
      bitnames+=*techbitnameIt;
    }
    std::cout<<"\tbitnames "<<bitnames<<std::endl;
    trgrundata.bitzeroname=technames[4];
    trgrundata.bitnames=bitnames;
    trgrundata.entry_id=revisionDML.getEntryInBranchByName(lumisession->nominalSchema(),lumi::LumiNames::trgdataTableName(),runnumberStr,branch_name);
    if(trgrundata.entry_id==0){
      revisionDML.bookNewEntry(lumisession->nominalSchema(),LumiNames::trgdataTableName(),trgrundata);
      std::cout<<"trgrundata revision_id "<<trgrundata.revision_id<<" entry_id "<<trgrundata.entry_id<<" data_id "<<trgrundata.data_id<<std::endl;
      revisionDML.addEntry(lumisession->nominalSchema(),LumiNames::trgdataTableName(),trgrundata,branch_id,branch_name);
  }else{
      revisionDML.bookNewRevision(lumisession->nominalSchema(),LumiNames::trgdataTableName(),trgrundata);
      std::cout<<"trgrundata revision_id "<<trgrundata.revision_id<<" entry_id "<<trgrundata.entry_id<<" data_id "<<trgrundata.data_id<<std::endl;
      revisionDML.addRevision(lumisession->nominalSchema(),LumiNames::trgdataTableName(),trgrundata,branch_id,branch_name);
    }
    std::cout<<"inserting trgrundata "<<std::endl;
    revisionDML.insertTrgRunData(lumisession->nominalSchema(),trgrundata);
    std::cout<<"inserting lstrg data"<<std::endl;
 
    unsigned int trglscount=0;   
    // trglscount=0;
    coral::IBulkOperation* lstrgInserter=0; 
    unsigned int comittedls=0;
    for(deadIt=deadtimesBeg;deadIt!=deadtimesEnd;++deadIt,++trglscount ){
      unsigned int cmslscount=trglscount+1;
      if(!lumisession->transaction().isActive()){ 
	lumisession->transaction().start(false);
	coral::ITable& lstrgtable=lumisession->nominalSchema().tableHandle(LumiNames::lstrgTableName());
	lstrgInserter=lstrgtable.dataEditor().bulkInsert(lstrgData,2048);
      }else{
	if(deadIt==deadtimesBeg){
	  coral::ITable& lstrgtable=lumisession->nominalSchema().tableHandle(LumiNames::lstrgTableName());
	  lstrgInserter=lstrgtable.dataEditor().bulkInsert(lstrgData,2048);
	}
      }
      data_id = trgrundata.data_id;
      trgrunnum = irunnumber;
      cmslsnum = cmslscount;
      deadtime = *deadIt;
      deadfrac = deadfracs[trglscount];
      //bitzerocount = algocounts[trglscount][0];//use algobit_0
      //bitzeroprescale = prescalealgo[cmslsnum][0];
      bitzerocount=techcounts[trglscount][4]; //use techbit_4
      bitzeroprescale=prescaletech[cmslsnum][4];
      std::vector<unsigned int> fullprescales;
      fullprescales.reserve(prescalealgo[cmslsnum].size()+prescaletech[cmslsnum].size());
      fullprescales.insert(fullprescales.end(),prescalealgo[cmslsnum].begin(),prescalealgo[cmslsnum].end());
      fullprescales.insert(fullprescales.end(),prescaletech[cmslsnum].begin(),prescaletech[cmslsnum].end());

      prescaleblob.resize(sizeof(unsigned int)*(fullprescales.size()));
      void* prescaleblob_StartAddress = prescaleblob.startingAddress();
      std::memmove(prescaleblob_StartAddress,&fullprescales[0],sizeof(unsigned int)*(fullprescales.size()));
      
      std::vector<unsigned int> fullcounts;
      fullcounts.reserve(algocounts[trglscount].size()+techcounts[trglscount].size());
      fullcounts.insert(fullcounts.end(),algocounts[trglscount].begin(),algocounts[trglscount].end());
      fullcounts.insert(fullcounts.end(),techcounts[trglscount].begin(),techcounts[trglscount].end());
      trgcountblob.resize(sizeof(unsigned int)*(fullcounts.size()));
      void* trgcountblob_StartAddress = trgcountblob.startingAddress();
      std::memmove(trgcountblob_StartAddress,&fullcounts[0],sizeof(unsigned int)*(fullcounts.size()));

      lstrgInserter->processNextIteration();	
      lstrgInserter->flush();
      ++comittedls;
      if(comittedls==commitintv){
	std::cout<<"\t committing in LS chunck "<<comittedls<<std::endl; 
	delete lstrgInserter; lstrgInserter=0;
	lumisession->transaction().commit();
	comittedls=0;
	std::cout<<"\t committed "<<std::endl; 
      }else if( trglscount==( totalcmsls-1) ){
	std::cout<<"\t committing at the end"<<std::endl; 
	delete lstrgInserter; lstrgInserter=0;
	lumisession->transaction().commit();
	std::cout<<"\t done"<<std::endl; 
      }
    }

    return trgrundata.data_id;
  }
  const std::string TRGScalers2DB::dataType() const{
    return "TRG";
  }
  const std::string TRGScalers2DB::sourceType() const{
    return "DB";
  }
  //utilities
  std::string TRGScalers2DB::int2str(unsigned int t, unsigned int width){
    std::stringstream ss;
    ss.width(width);
    ss.fill('0');
    ss<<t;
    return ss.str();
  }
  unsigned int TRGScalers2DB::str2int(const std::string& s){
    std::istringstream myStream(s);
    unsigned int i;
    if(myStream>>i){
      return i;
    }else{
      throw lumi::Exception(std::string("str2int error"),"str2int","TRGScalers2DB");
    }
  }
  TRGScalers2DB::~TRGScalers2DB(){}
}//ns lumi
#include "RecoLuminosity/LumiProducer/interface/DataPipeFactory.h"
DEFINE_EDM_PLUGIN(lumi::DataPipeFactory,lumi::TRGScalers2DB,"TRGScalers2DB");
#endif

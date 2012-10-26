#ifndef RecoLuminosity_LumiProducer_CMSRunSummary2DB_h 
#define RecoLuminosity_LumiProducer_CMSRunSummary2DB_h 
#include "CoralBase/AttributeList.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/AttributeSpecification.h"
#include "CoralBase/Exception.h"
#include "CoralBase/TimeStamp.h"
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
#include "RecoLuminosity/LumiProducer/interface/DataPipe.h"
#include "RecoLuminosity/LumiProducer/interface/LumiNames.h"
#include "RecoLuminosity/LumiProducer/interface/idDealer.h"
#include "RecoLuminosity/LumiProducer/interface/Exception.h"
#include "RecoLuminosity/LumiProducer/interface/DBConfig.h"
#include "RecoLuminosity/LumiProducer/interface/ConstantDef.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <string>
#include <boost/regex.hpp>
#include <boost/tokenizer.hpp>

namespace lumi{
  class CMSRunSummary2DB : public DataPipe{
  public:
    CMSRunSummary2DB( const std::string& dest);
    virtual unsigned long long retrieveData( unsigned int runnumber );
    virtual const std::string dataType() const;
    virtual const std::string sourceType() const;
    unsigned int str2int(const std::string& s) const;
    virtual ~CMSRunSummary2DB();
  private:
    struct cmsrunsum{
      std::string l1key;
      std::string amodetag;	      
      int egev;
      std::string hltkey;
      std::string fillnumber; //convert to number when write into lumi
      std::string sequence;
      std::string fillscheme;
      int ncollidingbunches;      
      coral::TimeStamp startT;
      coral::TimeStamp stopT;
    };
    bool isCollisionRun(const lumi::CMSRunSummary2DB::cmsrunsum& rundata);
    void parseFillCSV(const std::string& csvsource, cmsrunsum& result);
  };//cl CMSRunSummary2DB
  //
  //implementation
  //
  void
  CMSRunSummary2DB::parseFillCSV(const std::string& csvsource, lumi::CMSRunSummary2DB::cmsrunsum& result){
    result.fillscheme=std::string("");
    result.ncollidingbunches=0;
    std::ifstream csvfile;
    csvfile.open(csvsource.c_str());
    if(!csvfile){
      std::cout<<"[warning] unable to open file: "<<csvsource<<std::endl;
      return;
    }
    typedef boost::tokenizer< boost::escaped_list_separator<char> > Tokenizer;
    std::vector<std::string> record;
    std::string line;
    while(std::getline(csvfile,line)){
      Tokenizer tok(line);
      record.assign(tok.begin(),tok.end());
      if(record.size()<3) continue;
      std::string fillnum=record[0];
      if(fillnum==result.fillnumber){
	result.fillscheme=record[1];
	std::string ncollidingbunchesStr=record[2];
	result.ncollidingbunches=str2int(ncollidingbunchesStr);
	break;
      }
    }
  }
  CMSRunSummary2DB::CMSRunSummary2DB(const std::string& dest):DataPipe(dest){}
  bool CMSRunSummary2DB::isCollisionRun(const  lumi::CMSRunSummary2DB::cmsrunsum& rundata){
    bool isCollision=false;
    bool isPhysics=false;
    std::string hk=rundata.hltkey;
    std::string lk=rundata.l1key;
    boost::match_results<std::string::const_iterator> what;
    const boost::regex lexpr("^TSC_.+_collisions_.+");
    boost::regex_match(lk,what,lexpr,boost::match_default);
    if(what[0].matched) isCollision=true;
    const boost::regex hexpr("^/cdaq/physics/.+");
    boost::regex_match(hk,what,hexpr,boost::match_default);
    if(what[0].matched) isPhysics=true;
    return (isCollision&&isPhysics);
  }
  unsigned long long 
  CMSRunSummary2DB::retrieveData( unsigned int runnumber){
    /**
       //select distinct name from runsession_parameter
       l1key: select string_value from cms_runinfo.runsession_parameter where runnumber=:runnumber and name='CMS.TRG:TSC_KEY';
       amodetag: select distinct(string_value),session_id from cms_runinfo.runsession_parameter where runnumber=:runnumber and name='CMS.SCAL:AMODEtag' 
       egev: select distinct(string_value) from cms_runinfo.runsession_parameter where runnumber=:runnumber and name='CMS.SCAL:EGEV'
       hltkey: select string_value from cms_runinfo.runsession_parameter where runnumber=129265 and name='CMS.LVL0:HLT_KEY_DESCRIPTION';
       fillnumber: select string_value from cms_runinfo.runsession_parameter where runnumber=129265 and name='CMS.SCAL:FILLN' order by time;//take the first one       sequence: select string_value from cms_runinfo.runsession_parameter where runnumber=129265 and name='CMS.LVL0:SEQ_NAME'
    **/
    cmsrunsum result;
    std::string runinfoschema("CMS_RUNINFO");
    std::string gtmonschema("CMS_GT_MON");
    std::string runsessionParamTable("RUNSESSION_PARAMETER");
    std::string globalrunTable("GLOBAL_RUNS");
    coral::ConnectionService* svc=new coral::ConnectionService;
    lumi::DBConfig dbconf(*svc);
    if(!m_authpath.empty()){
      dbconf.setAuthentication(m_authpath);
    }
    
    //std::cout<<"m_source "<<m_source<<std::endl;
    std::string::size_type cutpos=m_source.find(';');
    std::string dbsource=m_source;
    std::string csvsource("");
    if(cutpos!=std::string::npos){
      dbsource=m_source.substr(0,cutpos);
      csvsource=m_source.substr(cutpos+1);
    }
    //std::cout<<"dbsource: "<<dbsource<<" , csvsource: "<<csvsource<<std::endl;
    coral::ISessionProxy* runinfosession=svc->connect(dbsource,coral::ReadOnly);
    try{
      coral::ITypeConverter& tpc=runinfosession->typeConverter();
      tpc.setCppTypeForSqlType("unsigned int","NUMBER(38)");
      runinfosession->transaction().start(true);
      coral::ISchema& runinfoschemaHandle=runinfosession->schema(runinfoschema);
      if(!runinfoschemaHandle.existsTable(runsessionParamTable)){
	throw lumi::Exception(std::string("non-existing table "+runsessionParamTable),"CMSRunSummary2DB","retrieveData");
      }
      coral::IQuery* amodetagQuery=runinfoschemaHandle.tableHandle(runsessionParamTable).newQuery();
      coral::AttributeList amodetagOutput;
      amodetagOutput.extend("amodetag",typeid(std::string));
      coral::AttributeList amodetagCondition;
      amodetagCondition=coral::AttributeList();
      amodetagCondition.extend("name",typeid(std::string));
      amodetagCondition.extend("runnumber",typeid(unsigned int));
      amodetagCondition["name"].data<std::string>()=std::string("CMS.SCAL:AMODEtag");
      amodetagCondition["runnumber"].data<unsigned int>()=runnumber;
      amodetagQuery->addToOutputList("distinct(STRING_VALUE)");
      amodetagQuery->setCondition("NAME=:name AND RUNNUMBER=:runnumber",amodetagCondition);
      //amodetagQuery->limitReturnedRows(1);
      amodetagQuery->defineOutput(amodetagOutput);
      coral::ICursor& amodetagCursor=amodetagQuery->execute();
      std::vector<std::string> amodes;
      while (amodetagCursor.next()){
	const coral::AttributeList& row=amodetagCursor.currentRow();
	amodes.push_back(row["amodetag"].data<std::string>());
	//result.amodetag=row["amodetag"].data<std::string>();
      }
      //
      //priority pick the one contains PHYS if not found pick the first
      //
      std::string amd;
      for(std::vector<std::string>::iterator it=amodes.begin();it!=amodes.end();++it){
	if(it->find("PHYS")==std::string::npos) continue;
	amd=*it;
      }
      if(amd.size()==0&&amodes.size()!=0){
	amd=*(amodes.begin());
      }
      if(amd.size()==0){
	 amd=std::string("PROTPHYS");//last resort
      }
      //std::cout<<"amd "<<amd<<std::endl;
      result.amodetag=amd;
      delete amodetagQuery;
      
      coral::IQuery* egevQuery=runinfoschemaHandle.tableHandle(runsessionParamTable).newQuery();
      coral::AttributeList egevOutput;
      egevOutput.extend("egev",typeid(std::string));
      coral::AttributeList egevCondition;
      egevCondition=coral::AttributeList();
      egevCondition.extend("name",typeid(std::string));
      egevCondition.extend("runnumber",typeid(unsigned int));
      egevCondition["name"].data<std::string>()=std::string("CMS.SCAL:EGEV");
      egevCondition["runnumber"].data<unsigned int>()=runnumber;
      egevQuery->addToOutputList("distinct(STRING_VALUE)");
      egevQuery->setCondition("NAME=:name AND RUNNUMBER=:runnumber",egevCondition);
      egevQuery->defineOutput(egevOutput);
      coral::ICursor& egevCursor=egevQuery->execute();
      result.egev=0;
      while (egevCursor.next()){
	const coral::AttributeList& row=egevCursor.currentRow();
	std::string egevstr=row["egev"].data<std::string>();
	int tmpgev=str2int(egevstr);
	if(tmpgev>result.egev){
	  result.egev=tmpgev;
	}
      }
      if(result.egev==0){
	 result.egev=3500;//last resort
      }
      delete egevQuery;
      
      coral::IQuery* seqQuery=runinfoschemaHandle.tableHandle(runsessionParamTable).newQuery();
      coral::AttributeList seqBindVariableList;
      seqBindVariableList.extend("runnumber",typeid(unsigned int));
      seqBindVariableList.extend("name",typeid(std::string));
      
      seqBindVariableList["runnumber"].data<unsigned int>()=runnumber;
      seqBindVariableList["name"].data<std::string>()=std::string("CMS.LVL0:SEQ_NAME");
      seqQuery->setCondition("RUNNUMBER=:runnumber AND NAME=:name",seqBindVariableList);
      seqQuery->addToOutputList("STRING_VALUE");
      coral::ICursor& seqCursor=seqQuery->execute();
      
      while( seqCursor.next() ){
	const coral::AttributeList& row=seqCursor.currentRow();
	result.sequence=row["STRING_VALUE"].data<std::string>();
      }
      delete seqQuery;
      
      coral::IQuery* hltkeyQuery=runinfoschemaHandle.tableHandle(runsessionParamTable).newQuery();
      coral::AttributeList hltkeyBindVariableList;
      hltkeyBindVariableList.extend("runnumber",typeid(unsigned int));
      hltkeyBindVariableList.extend("name",typeid(std::string));

      hltkeyBindVariableList["runnumber"].data<unsigned int>()=runnumber;
      hltkeyBindVariableList["name"].data<std::string>()=std::string("CMS.LVL0:HLT_KEY_DESCRIPTION");
      hltkeyQuery->setCondition("RUNNUMBER =:runnumber AND NAME =:name",hltkeyBindVariableList);
      hltkeyQuery->addToOutputList("STRING_VALUE");   
      coral::ICursor& hltkeyCursor=hltkeyQuery->execute();
      
    while( hltkeyCursor.next() ){
      const coral::AttributeList& row=hltkeyCursor.currentRow();
      result.hltkey=row["STRING_VALUE"].data<std::string>();
    }
    delete hltkeyQuery;
    
    coral::IQuery* fillQuery=runinfoschemaHandle.tableHandle(runsessionParamTable).newQuery();
    coral::AttributeList fillBindVariableList;
    fillBindVariableList.extend("runnumber",typeid(unsigned int));
    fillBindVariableList.extend("name",typeid(std::string));
    
    fillBindVariableList["runnumber"].data<unsigned int>()=runnumber;
    fillBindVariableList["name"].data<std::string>()=std::string("CMS.SCAL:FILLN");
    fillQuery->setCondition("RUNNUMBER=:runnumber AND NAME=:name",fillBindVariableList);
    fillQuery->addToOutputList("STRING_VALUE"); 
    fillQuery->addToOrderList("TIME");
    //fillQuery->limitReturnedRows(1);
    coral::ICursor& fillCursor=fillQuery->execute();
    unsigned int cc=0;
    while( fillCursor.next() ){
      const coral::AttributeList& row=fillCursor.currentRow();
      if (cc==0){
	result.fillnumber=row["STRING_VALUE"].data<std::string>();
      }
      ++cc;
    }
    delete fillQuery;
    if (result.fillnumber.empty()){
      throw nonCollisionException("retrieveData","CMSRunSummary2DB");
    }
    coral::IQuery* l1keyQuery=runinfoschemaHandle.tableHandle(runsessionParamTable).newQuery();
    coral::AttributeList l1keyOutput;
    l1keyOutput.extend("l1key",typeid(std::string));
    coral::AttributeList l1keyCondition;
    l1keyCondition=coral::AttributeList();
    l1keyCondition.extend("name",typeid(std::string));
    l1keyCondition.extend("runnumber",typeid(unsigned int));
    l1keyCondition["name"].data<std::string>()=std::string("CMS.TRG:TSC_KEY");
    l1keyCondition["runnumber"].data<unsigned int>()=runnumber;
    l1keyQuery->addToOutputList("STRING_VALUE");
    l1keyQuery->setCondition("NAME=:name AND RUNNUMBER=:runnumber",l1keyCondition);
    //l1keyQuery->limitReturnedRows(1);
    l1keyQuery->defineOutput(l1keyOutput);
    coral::ICursor& l1keyCursor=l1keyQuery->execute();
    while (l1keyCursor.next()){
      const coral::AttributeList& row=l1keyCursor.currentRow();
      result.l1key=row["l1key"].data<std::string>();
    }
    delete l1keyQuery;
    /**
       start/stop time:
       select start_time,stop_time from cms_gt_mon.global_runs where run_number=:runnum
    **/
    coral::ISchema& gtmonschemaHandle=runinfosession->schema(gtmonschema);
    if(!gtmonschemaHandle.existsTable(globalrunTable)){
      throw lumi::Exception(std::string("non-existing table "+globalrunTable),"CMSRunSummary2DB","retrieveData");
    }
    coral::IQuery* startTQuery=gtmonschemaHandle.tableHandle(globalrunTable).newQuery();
    coral::AttributeList startTVariableList;
    startTVariableList.extend("runnum",typeid(unsigned int));
    startTQuery->setCondition("RUN_NUMBER=:runnum",startTVariableList);
    startTVariableList["runnum"].data<unsigned int>()=runnumber;
    startTQuery->addToOutputList("START_TIME");
    startTQuery->addToOutputList("STOP_TIME");
    coral::ICursor& startTCursor=startTQuery->execute();
    unsigned int rowcounter=0;
    while( startTCursor.next() ){
      const coral::AttributeList& row=startTCursor.currentRow();
      result.startT=row["START_TIME"].data<coral::TimeStamp>();
      result.stopT=row["STOP_TIME"].data<coral::TimeStamp>();	
	++rowcounter;
    }
    delete startTQuery;
    if(rowcounter==0){//fallback to runinfo if gt has no data
      coral::IQuery* runstartTQuery=runinfoschemaHandle.tableHandle(runsessionParamTable).newQuery();
      coral::AttributeList runstartTVariableList;
      runstartTVariableList.extend("runnumber",typeid(unsigned int));
      runstartTVariableList.extend("name",typeid(std::string));
      
      runstartTVariableList["runnumber"].data<unsigned int>()=runnumber;
      runstartTVariableList["name"].data<std::string>()=std::string("CMS.LVL0:START_TIME_T");
      runstartTQuery->setCondition("RUNNUMBER=:runnumber AND NAME=:name",runstartTVariableList);
      runstartTQuery->addToOutputList("TIME"); 
      coral::ICursor& runstartTCursor=runstartTQuery->execute();
      
      while( runstartTCursor.next() ){
	const coral::AttributeList& row=runstartTCursor.currentRow();
	result.startT=row["TIME"].data<coral::TimeStamp>();
      }
      delete runstartTQuery;
      coral::IQuery* runstopTQuery=runinfoschemaHandle.tableHandle(runsessionParamTable).newQuery();
      coral::AttributeList runstopTVariableList;
      runstopTVariableList.extend("runnumber",typeid(unsigned int));
      runstopTVariableList.extend("name",typeid(std::string));
      runstopTVariableList["runnumber"].data<unsigned int>()=runnumber;
      runstopTVariableList["name"].data<std::string>()=std::string("CMS.LVL0:STOP_TIME_T");
      runstopTQuery->setCondition("RUNNUMBER=:runnumber AND NAME=:name",runstopTVariableList);
      runstopTQuery->addToOutputList("TIME"); 
      coral::ICursor& runstopTCursor=runstopTQuery->execute();	 
      while( runstopTCursor.next() ){
	const coral::AttributeList& row=runstopTCursor.currentRow();
	result.stopT=row["TIME"].data<coral::TimeStamp>();
      }
      delete runstopTQuery;
      }
    }catch( const coral::Exception& er){
      runinfosession->transaction().rollback();
      delete runinfosession;
      delete svc;
      throw er;
    }

    runinfosession->transaction().commit();
    delete runinfosession;
    
    if(csvsource.size()!=0){
      parseFillCSV(csvsource,result);
    }else{
      result.fillscheme=std::string("");
      result.ncollidingbunches=0;
    }
    std::cout<<"result for run "<<runnumber<<" : sequence : "<<result.sequence<<" : hltkey : "<<result.hltkey<<" : fillnumber : "<<result.fillnumber<<" : l1key : "<<result.l1key<<" : amodetag :"<<result.amodetag<<" : egev : "<<result.egev<<" : fillscheme "<<result.fillscheme<<" : ncollidingbunches : "<<result.ncollidingbunches<<" start : "<<result.startT.toString()<<" stop : "<<result.stopT.toString()<<std::endl; 

    //std::cout<<"connecting to dest "<<m_dest<<std::endl; 
    coral::ISessionProxy* destsession=svc->connect(m_dest,coral::Update);
    
    coral::ITypeConverter& desttpc=destsession->typeConverter();
    desttpc.setCppTypeForSqlType("unsigned int","NUMBER(10)");
    try{
      destsession->transaction().start(false);
      coral::ISchema& destschema=destsession->nominalSchema();
      coral::ITable& destruntable=destschema.tableHandle(LumiNames::cmsrunsummaryTableName());
      coral::AttributeList runData;
      destruntable.dataEditor().rowBuffer(runData);
      runData["RUNNUM"].data<unsigned int>()=runnumber;
      runData["FILLNUM"].data<unsigned int>()=str2int(result.fillnumber);
      runData["SEQUENCE"].data<std::string>()=result.sequence;
      runData["HLTKEY"].data<std::string>()=result.hltkey;
      runData["STARTTIME"].data<coral::TimeStamp>()=result.startT;
      runData["STOPTIME"].data<coral::TimeStamp>()=result.stopT;
      runData["AMODETAG"].data<std::string>()=result.amodetag;
      runData["EGEV"].data<unsigned int>()=(unsigned int)result.egev;
      runData["L1KEY"].data<std::string>()=result.l1key;
      runData["FILLSCHEME"].data<std::string>()=result.fillscheme;
      runData["NCOLLIDINGBUNCHES"].data<unsigned int>()=result.ncollidingbunches;
      destruntable.dataEditor().insertRow(runData);
    }catch( const coral::Exception& er){
      std::cout<<"database problem "<<er.what()<<std::endl;
      destsession->transaction().rollback();
      delete destsession;
      delete svc;
      throw er;
    }
    destsession->transaction().commit();
    delete svc;
    return 0;
  }
  const std::string CMSRunSummary2DB::dataType() const{
    return "CMSRUNSUMMARY";
  }
  const std::string CMSRunSummary2DB::sourceType() const{
    return "DB";
  }
  unsigned int CMSRunSummary2DB::str2int(const std::string& s)  const{
    std::istringstream myStream(s);
    unsigned int i;
    if(myStream>>i){
      return i;
    }else{
      throw lumi::Exception(std::string("str2int error"),"str2int","CMSRunSummary2DB");
    }
  }
  CMSRunSummary2DB::~CMSRunSummary2DB(){}
}//ns lumi
#include "RecoLuminosity/LumiProducer/interface/DataPipeFactory.h"
DEFINE_EDM_PLUGIN(lumi::DataPipeFactory,lumi::CMSRunSummary2DB,"CMSRunSummary2DB");
#endif

#ifndef RecoLuminosity_LumiProducer_HLTV32DB_H 
#define RecoLuminosity_LumiProducer_HLTV32DB_H
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
#include <map>
#include <vector>
#include <string>

#include "RecoLuminosity/LumiProducer/interface/DataPipe.h"
namespace lumi{
  class HLTV32DB : public DataPipe{
    
  public:
    const static unsigned int COMMITLSINTERVAL=200; //commit interval in LS,totalrow=nls*(~200)
    explicit HLTV32DB(const std::string& dest);
    virtual void retrieveData( unsigned int );
    virtual const std::string dataType() const;
    virtual const std::string sourceType() const;
    virtual ~HLTV32DB();
    std::map< unsigned int, std::string > hltpathmap;
    struct hltinfo{
      unsigned int cmsluminr;
      std::string pathname;
      unsigned int hltinput;
      unsigned int hltaccept;
      unsigned int prescale;
    };
  };//cl HLTV32DB
  //
  //implementation
  //
  
  HLTV32DB::HLTV32DB(const std::string& dest):DataPipe(dest){}
  void HLTV32DB::retrieveData( unsigned int runnumber){
    std::string confdbschema("CMS_HLT");
    std::string hltschema("CMS_RUNINFO");
    std::string confdbpathtabname("PATHS");
    std::string triggerpathtabname("HLT_SUPERVISOR_TRIGGERPATHS");
    std::string lstabname("HLT_SUPERVISOR_LUMISECTIONS_V3");
    std::string maptabname("HLT_SUPERVISOR_SCALAR_MAP_V2");
    
    coral::ConnectionService* svc=new coral::ConnectionService;
    lumi::DBConfig dbconf(*svc);
    if(!m_authpath.empty()){
      dbconf.setAuthentication(m_authpath);
    }
    /**retrieve hlt info with 3 queries from runinfo
       1. select distinct ( PATHID ) from HLT_SUPERVISOR_TRIGGERPATHS where runnumber=158878;
       2. retrieve hltpath map with n query from cms_hlt (n=number of pathids)
          select name from cms_hlt.paths where pathid=:pathid
       3. select count(*) from HLT_SUPERVISOR_LUMISECTIONS_V3 where runnumber=158878;
       4. select tr.lsnumber,tr.pathid,tr.l1pass,tr.paccept,ls.psindex,sm.psvalue from hlt_supervisor_triggerpaths tr,hlt_supervisor_lumisections_v3 ls,hlt_supervisor_scalar_map_v2 sm where tr.runnumber=ls.runnumber and tr.lsnumber=ls.lsnumber and sm.runnumber=tr.runnumber and sm.pathid=tr.pathid and sm.psindex=ls.psindex and tr.runnumber=157805 order by tr.lsnumber;
    **/
    
    //std::cout<<"m_source "<<m_source<<std::endl;
    coral::ISessionProxy* srcsession=svc->connect(m_source, coral::ReadOnly);
    coral::ITypeConverter& tpc=srcsession->typeConverter();
    tpc.setCppTypeForSqlType("unsigned int","NUMBER(11)");
    srcsession->transaction().start(true);
    coral::ISchema& hltSchemaHandle=srcsession->schema(hltschema);
    coral::ISchema& confdbSchemaHandle=srcsession->schema(confdbschema);
    if( !hltSchemaHandle.existsTable(triggerpathtabname) || !hltSchemaHandle.existsTable(lstabname) || !hltSchemaHandle.existsTable(maptabname) ){
      throw lumi::Exception("missing hlt tables" ,"retrieveData","HLTV32DB");
    }

    coral::AttributeList bindVariableList;
    bindVariableList.extend("runnumber",typeid(unsigned int));
    bindVariableList["runnumber"].data<unsigned int>()=runnumber;
    coral::IQuery* q1=hltSchemaHandle.tableHandle(triggerpathtabname).newQuery();
    coral::AttributeList hltpathid;
    hltpathid.extend("hltpathid",typeid(unsigned int));
    q1->addToOutputList("distinct PATHID","hltpathid");
    q1->setCondition("RUNNUMBER =:runnumber",bindVariableList);
    q1->defineOutput(hltpathid);
    coral::ICursor& c=q1->execute();
    while( c.next() ){
      unsigned int hid=c.currentRow()["hltpathid"].data<unsigned int>();
      hltpathmap.insert(std::make_pair(hid,""));
    }
    delete q1;
    //std::cout<<"npath "<<npath<<std::endl;
    std::map< unsigned int, std::string >::iterator mpit;
    std::map< unsigned int, std::string >::iterator mpitBeg=hltpathmap.begin();
    std::map< unsigned int, std::string >::iterator mpitEnd=hltpathmap.end();
    for( mpit=mpitBeg;mpit!=mpitEnd;++mpit){
      coral::IQuery* mq=confdbSchemaHandle.newQuery();
      coral::AttributeList mqbindVariableList;
      mqbindVariableList.extend("pathid",typeid(unsigned int));
      mqbindVariableList["pathid"].data<unsigned int>()=mpit->first;
      mq->addToTableList(confdbpathtabname);
      mq->addToOutputList("NAME","hltpathname");
      mq->setCondition("PATHID=:pathid",mqbindVariableList);
      coral::ICursor& mqcursor=mq->execute();
      while( mqcursor.next() ){
	std::string pathname=mqcursor.currentRow()["hltpathname"].data<std::string>();
	hltpathmap[mpit->first]=pathname;
      }
      delete mq;
    }
    //for( mpit=mpitBeg;mpit!=mpitEnd;++mpit){
    //  std::cout<<mpit->first<<" "<<mpit->second<<std::endl;
    //}
    unsigned int nls=0;
    
    coral::IQuery* nq=hltSchemaHandle.tableHandle(lstabname).newQuery();
    coral::AttributeList nqbindVariableList;
    coral::AttributeList nqout;
    nqout.extend("nls",typeid(unsigned int));
    nqbindVariableList.extend("runnumber",typeid(unsigned int));
    nqbindVariableList["runnumber"].data<unsigned int>()=runnumber;
    nq->addToOutputList("count(*)","nls");
    nq->setCondition("RUNNUMBER =:runnumber",nqbindVariableList);    
    nq->defineOutput(nqout);
    coral::ICursor& nqcursor=nq->execute();
    while( nqcursor.next() ){
      nls=nqcursor.currentRow()["nls"].data<unsigned int>();
    }
    delete nq;
    //std::cout<<"nls "<<nls<<std::endl; 
    std::vector< std::map<unsigned int , HLTV32DB::hltinfo> > hltresult;
    hltresult.reserve(nls);
    //fix all size
    for(unsigned int i=1;i<=nls;++i){
      std::map<unsigned int, HLTV32DB::hltinfo> allpaths;
      std::map<unsigned int, std::string>::iterator aIt;
      std::map<unsigned int, std::string>::iterator aItBeg=hltpathmap.begin();
      std::map<unsigned int, std::string>::iterator aItEnd=hltpathmap.end();
      for(aIt=aItBeg;aIt!=aItEnd;++aIt){
	HLTV32DB::hltinfo ct;
	ct.cmsluminr=i;
        ct.pathname=aIt->second;
        ct.hltinput=0;
        ct.hltaccept=0;
        ct.prescale=0;	
	allpaths.insert(std::make_pair(aIt->first,ct));
      }
      hltresult.push_back(allpaths);
    }
    coral::IQuery* jq=hltSchemaHandle.newQuery();
    coral::AttributeList jqbindVariableList;
    jqbindVariableList.extend("runnumber",typeid(unsigned int));
    jqbindVariableList["runnumber"].data<unsigned int>()=runnumber;
    coral::AttributeList jqoutput;
    jqoutput.extend("lsnumber",typeid(unsigned int));
    jqoutput.extend("pathid",typeid(unsigned int));
    jqoutput.extend("l1pass",typeid(unsigned int));
    jqoutput.extend("paccept",typeid(unsigned int));
    jqoutput.extend("psvalue",typeid(unsigned int));
    jq->addToTableList(triggerpathtabname,"tr");
    jq->addToTableList(lstabname,"ls");
    jq->addToTableList(maptabname,"sm");
    jq->addToOutputList("tr.LSNUMBER","lsnumber");
    jq->addToOutputList("tr.PATHID","pathid");
    jq->addToOutputList("tr.L1PASS","l1pass");
    jq->addToOutputList("tr.PACCEPT","paccept");
    jq->addToOutputList("sm.PSVALUE","psvalue");
    jq->setCondition("tr.RUNNUMBER=ls.RUNNUMBER AND tr.LSNUMBER=ls.LSNUMBER and sm.RUNNUMBER=tr.RUNNUMBER and sm.PATHID=tr.PATHID and sm.PSINDEX=ls.PSINDEX and tr.RUNNUMBER=:runnumber",jqbindVariableList);  
    jq->defineOutput(jqoutput);
    jq->addToOrderList("tr.LSNUMBER");
    jq->setRowCacheSize(10692);
    coral::ICursor& jqcursor=jq->execute();
    
    while( jqcursor.next() ){
      const coral::AttributeList& row=jqcursor.currentRow();
      unsigned int currentLumiSection=row["lsnumber"].data<unsigned int>();
      std::map<unsigned int,hltinfo>& allpathinfo=hltresult.at(currentLumiSection-1);
      unsigned int pathid=row["pathid"].data<unsigned int>();
      hltinfo& pathcontent=allpathinfo[pathid];
      pathcontent.hltinput=row["l1pass"].data<unsigned int>();
      pathcontent.hltaccept=row["paccept"].data<unsigned int>();
      pathcontent.prescale=row["psvalue"].data<unsigned int>();
    }
    delete jq;
    srcsession->transaction().commit();
    delete srcsession;
    //std::cout<<"result size "<<hltresult.size()<<std::endl;
    //std::vector< std::map<unsigned int, HLTV32DB::hltinfo> >::iterator hltIt;
    //std::vector< std::map<unsigned int, HLTV32DB::hltinfo> >::iterator hltItBeg=hltresult.begin();
    //std::vector< std::map<unsigned int, HLTV32DB::hltinfo> >::iterator hltItEnd=hltresult.end();
    // Write into DB
    //

    unsigned int totalcmsls=hltresult.size();
    unsigned int npath=hltpathmap.size();
    std::cout<<"inserting totalhltls "<<totalcmsls<<" total path "<<npath<<std::endl;
    std::map< unsigned long long, std::vector<unsigned long long> > idallocationtable;
    coral::ISessionProxy* destsession=svc->connect(m_dest,coral::Update);
    coral::ITypeConverter& lumitpc=destsession->typeConverter();
    lumitpc.setCppTypeForSqlType("unsigned int","NUMBER(7)");
    lumitpc.setCppTypeForSqlType("unsigned int","NUMBER(10)");
    lumitpc.setCppTypeForSqlType("unsigned long long","NUMBER(20)");

    std::vector< std::map<unsigned int,HLTV32DB::hltinfo> >::iterator hltIt;
    std::vector< std::map<unsigned int,HLTV32DB::hltinfo> >::iterator hltItBeg=hltresult.begin();
    std::vector< std::map<unsigned int,HLTV32DB::hltinfo> >::iterator hltItEnd=hltresult.end();
    //for(hltIt=hltItBeg;hltIt!=hltItEnd;++hltIt){
    //  std::map<unsigned int,HLTV32DB::hltinfo>::iterator pathIt;
    //  std::map<unsigned int,HLTV32DB::hltinfo>::iterator pathItBeg=hltIt->begin();
    //  std::map<unsigned int,HLTV32DB::hltinfo>::iterator pathItEnd=hltIt->end();
    //  for(pathIt=pathItBeg;pathIt!=pathItEnd;++pathIt){
    //	std::cout<<"cmslsnr "<<pathIt->second.cmsluminr<<" "<<pathIt->second.pathname<<" "<<pathIt->second.hltinput<<" "<<pathIt->second.hltaccept<<" "<<pathIt->second.prescale<<std::endl;
    //  }
    //}
    try{
      std::cout<<"\t allocating total ids "<<totalcmsls*npath<<std::endl; 
      destsession->transaction().start(false);
      lumi::idDealer idg(destsession->nominalSchema());
      unsigned long long hltID = idg.generateNextIDForTable(LumiNames::hltTableName(),totalcmsls*npath)-totalcmsls*npath;
      destsession->transaction().commit();
      unsigned int hltlscount=0;
      for(hltIt=hltItBeg;hltIt!=hltItEnd;++hltIt,++hltlscount){
	std::vector<unsigned long long> pathvec;
	pathvec.reserve(npath);
	for(unsigned int i=0;i<npath;++i,++hltID){
	  pathvec.push_back(hltID);
	}
	idallocationtable.insert(std::make_pair(hltlscount,pathvec));
      }
      std::cout<<"\t all ids allocated"<<std::endl; 

      coral::AttributeList hltData;
      hltData.extend("HLT_ID",typeid(unsigned long long));
      hltData.extend("RUNNUM",typeid(unsigned int));
      hltData.extend("CMSLSNUM",typeid(unsigned int));
      hltData.extend("PATHNAME",typeid(std::string));
      hltData.extend("INPUTCOUNT",typeid(unsigned int));
      hltData.extend("ACCEPTCOUNT",typeid(unsigned int));
      hltData.extend("PRESCALE",typeid(unsigned int));
      
      //loop over lumi LS
      unsigned long long& hlt_id=hltData["HLT_ID"].data<unsigned long long>();
      unsigned int& hltrunnum=hltData["RUNNUM"].data<unsigned int>();
      unsigned int& cmslsnum=hltData["CMSLSNUM"].data<unsigned int>();
      std::string& pathname=hltData["PATHNAME"].data<std::string>();
      unsigned int& inputcount=hltData["INPUTCOUNT"].data<unsigned int>();
      unsigned int& acceptcount=hltData["ACCEPTCOUNT"].data<unsigned int>();
      unsigned int& prescale=hltData["PRESCALE"].data<unsigned int>();
      hltlscount=0;
      coral::IBulkOperation* hltInserter=0; 
      unsigned int comittedls=0;
      for(hltIt=hltItBeg;hltIt!=hltItEnd;++hltIt,++hltlscount){
	std::map<unsigned int,HLTV32DB::hltinfo>::const_iterator pathIt;
	std::map<unsigned int,HLTV32DB::hltinfo>::const_iterator pathBeg=hltIt->begin();
	std::map<unsigned int,HLTV32DB::hltinfo>::const_iterator pathEnd=hltIt->end();
	if(!destsession->transaction().isActive()){ 
	  destsession->transaction().start(false);
	  coral::ITable& hlttable=destsession->nominalSchema().tableHandle(LumiNames::hltTableName());
	  hltInserter=hlttable.dataEditor().bulkInsert(hltData,npath);
	}
	unsigned int hltpathcount=0;
	for(pathIt=pathBeg;pathIt!=pathEnd;++pathIt,++hltpathcount){
	  hlt_id = idallocationtable[hltlscount].at(hltpathcount);
	  hltrunnum = runnumber;
	  cmslsnum = pathIt->second.cmsluminr;
	  pathname = pathIt->second.pathname;
	  inputcount = pathIt->second.hltinput;
	  acceptcount = pathIt->second.hltaccept;
	  prescale = pathIt->second.prescale;
	  hltInserter->processNextIteration();
	}
	hltInserter->flush();
	++comittedls;
	if(comittedls==HLTV32DB::COMMITLSINTERVAL){
	  std::cout<<"\t committing in LS chunck "<<comittedls<<std::endl; 
	  delete hltInserter; hltInserter=0;
	  destsession->transaction().commit();
	  comittedls=0;
	  std::cout<<"\t committed "<<std::endl; 
	}else if( hltlscount==(totalcmsls-1) ){
	  std::cout<<"\t committing at the end"<<std::endl; 
	  delete hltInserter; hltInserter=0;
	  destsession->transaction().commit();
	  std::cout<<"\t done"<<std::endl; 
	}
      }
    }catch( const coral::Exception& er){
      std::cout<<"database problem "<<er.what()<<std::endl;
      destsession->transaction().rollback();
      delete destsession;
      delete svc;
      throw er;
    }
    delete destsession;
    delete svc;

  }
  const std::string HLTV32DB::dataType() const{
    return "HLTV3";
  }
  const std::string HLTV32DB::sourceType() const{
    return "DB";
  }
  HLTV32DB::~HLTV32DB(){}
}//ns lumi
#include "RecoLuminosity/LumiProducer/interface/DataPipeFactory.h"
DEFINE_EDM_PLUGIN(lumi::DataPipeFactory,lumi::HLTV32DB,"HLTV32DB");
#endif

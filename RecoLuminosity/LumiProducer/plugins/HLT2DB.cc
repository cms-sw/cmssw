#ifndef RecoLuminosity_LumiProducer_HLT2DB_H 
#define RecoLuminosity_LumiProducer_HLT2DB_H
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
  class HLT2DB : public DataPipe{
    
  public:
    const static unsigned int COMMITLSINTERVAL=200; //commit interval in LS,totalrow=nls*(~200)
    explicit HLT2DB(const std::string& dest);
    virtual void retrieveData( unsigned int );
    virtual const std::string dataType() const;
    virtual const std::string sourceType() const;
    virtual ~HLT2DB();

    struct hltinfo{
      unsigned int cmsluminr;
      std::string pathname;
      unsigned int hltinput;
      unsigned int hltaccept;
      unsigned int prescale;
      unsigned int hltconfigid;
    };
  };//cl HLT2DB
  //
  //implementation
  //
  HLT2DB::HLT2DB(const std::string& dest):DataPipe(dest){}
  void HLT2DB::retrieveData( unsigned int runnumber){
    
    std::string hltschema("CMS_RUNINFO");
    std::string tabname("HLT_SUPERVISOR_LUMISECTIONS_V2");
    std::string maptabname("HLT_SUPERVISOR_SCALAR_MAP");
    
    coral::ConnectionService* svc=new coral::ConnectionService;
    lumi::DBConfig dbconf(*svc);
    if(!m_authpath.empty()){
      dbconf.setAuthentication(m_authpath);
    }
    /**retrieve hlt info with 2 queries
       select count(distinct PATHNAME ) as npath from HLT_SUPERVISOR_LUMISECTIONS_V2 where runnr=110823 and lsnumber=1;
       select l.PATHNAME,l.LSNUMBER,l.L1PASS,l.PACCEPT,m.PSVALUE from hlt_supervisor_lumisections_v2 l, hlt_supervisor_scalar_map m where l.RUNNR=m.RUNNR and l.PSINDEX=m.PSINDEX and l.PATHNAME=m.PATHNAME and l.RUNNR=83037 order by l.LSNUMBER;
    **/
    
    //std::cout<<"m_source "<<m_source<<std::endl;
    coral::ISessionProxy* srcsession=svc->connect(m_source, coral::ReadOnly);
    coral::ITypeConverter& tpc=srcsession->typeConverter();
    tpc.setCppTypeForSqlType("unsigned int","NUMBER(11)");
    srcsession->transaction().start(true);
    coral::ISchema& hltSchemaHandle=srcsession->schema(hltschema);
    if( !hltSchemaHandle.existsTable(tabname) || !hltSchemaHandle.existsTable(maptabname) ){
      throw lumi::Exception("missing hlt tables" ,"retrieveData","HLT2DB");
    }
    std::vector< std::vector<HLT2DB::hltinfo> > hltresult;
    coral::AttributeList bindVariableList;
    bindVariableList.extend("runnumber",typeid(unsigned int));
    bindVariableList.extend("lsnumber",typeid(unsigned int));
    bindVariableList["runnumber"].data<unsigned int>()=runnumber;
    bindVariableList["lsnumber"].data<unsigned int>()=1;
    //coral::IQuery* q1=srcsession->nominalSchema().tableHandle(tabname).newQuery();
    //coral::AttributeList nls;
    //nls.extend("npath",typeid(unsigned int));
    //q1->addToOutputList("count(distinct PATHNAME)","npath");
    //q1->setCondition("RUNNR =:runnumber AND LSNUMBER =:lsnumber",bindVariableList);
    //q1->defineOutput(nls);
    //coral::ICursor& c=q1->execute();
    //if( !c.next() ){
    //c.close();
    // delete q1;
    //throw lumi::Exception("request run doen't exist","retrieveData","HLT2DB");
    //}else{
    //npath=c.currentRow()["npath"].data<unsigned int>();
    // c.close();
    //delete q1;
    //if(npath==0){
    //std::cout<<"request run is empty, do nothing"<<std::endl;
    //return;
    //}
    //}
    //std::cout<<"npath "<<npath<<std::endl;
    unsigned int npath=0;
    coral::IQuery* q2=srcsession->nominalSchema().newQuery();
    coral::AttributeList q2bindVariableList;
    q2bindVariableList.extend("runnumber",typeid(unsigned int));
    q2bindVariableList["runnumber"].data<unsigned int>()=runnumber;
    q2->addToTableList(tabname,"l");
    q2->addToTableList(maptabname,"m");
    q2->addToOutputList("l.LSNUMBER","lsnumber");
    q2->addToOutputList("l.PATHNAME","pathname");
    q2->addToOutputList("l.L1PASS","hltinput");
    q2->addToOutputList("l.PACCEPT","hltratecounter");
    q2->addToOutputList("m.PSVALUE","prescale");
    q2->addToOutputList("m.HLTKEY","hltconfigid");
    q2->setCondition("l.RUNNR=m.RUNNR and l.PSINDEX=m.PSINDEX and l.PATHNAME=m.PATHNAME and l.RUNNR =:runnumber",q2bindVariableList);   
    q2->addToOrderList("lsnumber");
    q2->setRowCacheSize(10692);
    coral::ICursor& cursor2=q2->execute();
    //unsigned int currentPath=0;
    unsigned int lastLumiSection=1;
    unsigned int currentLumiSection=0;
    // unsigned int counter=0;
    std::vector<hltinfo> allpaths;
    allpaths.reserve(200);
    while( cursor2.next() ){
      hltinfo pathcontent;
      const coral::AttributeList& row=cursor2.currentRow();
      currentLumiSection=row["lsnumber"].data<unsigned int>();
      pathcontent.cmsluminr=currentLumiSection;
      pathcontent.hltinput=row["hltinput"].data<unsigned int>();
      pathcontent.hltaccept=row["hltratecounter"].data<unsigned int>();
      pathcontent.pathname=row["pathname"].data<std::string>();
      pathcontent.prescale=row["prescale"].data<unsigned int>();
      pathcontent.hltconfigid=row["hltconfigid"].data<unsigned int>();
      if(currentLumiSection != lastLumiSection){
	hltresult.push_back(allpaths);
	//npath=allpaths.size();
	allpaths.clear();
      }
      lastLumiSection=currentLumiSection;
      allpaths.push_back(pathcontent);
      npath=allpaths.size();
    }
    hltresult.push_back(allpaths);
    cursor2.close();
    delete q2;
    srcsession->transaction().commit();
    delete srcsession;
    
    //
    // Write into DB
    //
    unsigned int totalcmsls=hltresult.size();
    std::cout<<"inserting totalhltls "<<totalcmsls<<std::endl;
    std::map< unsigned long long, std::vector<unsigned long long> > idallocationtable;
    coral::ISessionProxy* destsession=svc->connect(m_dest,coral::Update);
    coral::ITypeConverter& lumitpc=destsession->typeConverter();
    lumitpc.setCppTypeForSqlType("unsigned int","NUMBER(7)");
    lumitpc.setCppTypeForSqlType("unsigned int","NUMBER(10)");
    lumitpc.setCppTypeForSqlType("unsigned long long","NUMBER(20)");

    std::vector< std::vector<HLT2DB::hltinfo> >::const_iterator hltIt;
    std::vector< std::vector<HLT2DB::hltinfo> >::const_iterator hltBeg=hltresult.begin();
    std::vector< std::vector<HLT2DB::hltinfo> >::const_iterator hltEnd=hltresult.end();
    
    try{
       std::cout<<"\t allocating total ids "<<totalcmsls*npath<<std::endl; 
      destsession->transaction().start(false);
      lumi::idDealer idg(destsession->nominalSchema());
      unsigned long long hltID = idg.generateNextIDForTable(LumiNames::hltTableName(),totalcmsls*npath)-totalcmsls*npath;
      destsession->transaction().commit();
      unsigned int hltlscount=0;
      for(hltIt=hltBeg;hltIt!=hltEnd;++hltIt,++hltlscount){
	std::vector<unsigned long long> pathvec;
	pathvec.reserve(200);
	std::vector<HLT2DB::hltinfo>::const_iterator pathIt;
	std::vector<HLT2DB::hltinfo>::const_iterator pathBeg=hltIt->begin();
	std::vector<HLT2DB::hltinfo>::const_iterator pathEnd=hltIt->end();
	for(pathIt=pathBeg;pathIt!=pathEnd;++pathIt,++hltID){
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
      for(hltIt=hltBeg;hltIt!=hltEnd;++hltIt,++hltlscount){
	std::vector<HLT2DB::hltinfo>::const_iterator pathIt;
	std::vector<HLT2DB::hltinfo>::const_iterator pathBeg=hltIt->begin();
	std::vector<HLT2DB::hltinfo>::const_iterator pathEnd=hltIt->end();
	if(!destsession->transaction().isActive()){ 
	  destsession->transaction().start(false);
	  coral::ITable& hlttable=destsession->nominalSchema().tableHandle(LumiNames::hltTableName());
	  hltInserter=hlttable.dataEditor().bulkInsert(hltData,200);
	}
	unsigned int hltpathcount=0;
	for(pathIt=pathBeg;pathIt!=pathEnd;++pathIt,++hltpathcount){
	  hlt_id = idallocationtable[hltlscount].at(hltpathcount);
	  hltrunnum = runnumber;
	  cmslsnum = pathIt->cmsluminr;
	  pathname = pathIt->pathname;
	  inputcount = pathIt->hltinput;
	  acceptcount = pathIt->hltaccept;
	  prescale = pathIt->prescale;
	  hltInserter->processNextIteration();
	}
	hltInserter->flush();
	++comittedls;
	if(comittedls==HLT2DB::COMMITLSINTERVAL){
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
  const std::string HLT2DB::dataType() const{
    return "HLT";
  }
  const std::string HLT2DB::sourceType() const{
    return "DB";
  }
  HLT2DB::~HLT2DB(){}
}//ns lumi
#include "RecoLuminosity/LumiProducer/interface/DataPipeFactory.h"
DEFINE_EDM_PLUGIN(lumi::DataPipeFactory,lumi::HLT2DB,"HLT2DB");
#endif

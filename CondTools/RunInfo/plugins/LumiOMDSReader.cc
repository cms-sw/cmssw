#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondCore/DBCommon/interface/Time.h"
#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/ConnectionConfiguration.h"
#include "CondCore/DBCommon/interface/MessageLevel.h"
#include "CondCore/DBCommon/interface/Connection.h"
#include "CondCore/DBCommon/interface/CoralTransaction.h"
#include "CondCore/DBCommon/src/CoralConnectionProxy.h"
#include "CondFormats/RunInfo/interface/LuminosityInfo.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/AttributeSpecification.h"
#include "CoralBase/Exception.h"
#include "CoralBase/TimeStamp.h"
#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/ITypeConverter.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ICursor.h"
#include "RelationalAccess/ISchema.h"
#include "LumiOMDSReader.h"
#include "CondTools/RunInfo/interface/LumiReaderFactory.h"
//#include <iostream>
lumi::LumiOMDSReader::LumiOMDSReader(const edm::ParameterSet&pset):lumi::LumiReaderBase(pset),m_session(new cond::DBSession ){
  m_constr=pset.getParameter<std::string>("connect");
  std::string authPath=pset.getParameter<std::string>("authenticationPath");
  int messageLevel=pset.getUntrackedParameter<int>("messageLevel",0);
  switch (messageLevel) {
  case 0 :
    m_session->configuration().setMessageLevel( cond::Error );
    break;    
  case 1:
    m_session->configuration().setMessageLevel( cond::Warning );
    break;
  case 2:
    m_session->configuration().setMessageLevel( cond::Info );
    break;
  case 3:
    m_session->configuration().setMessageLevel( cond::Debug );
    break;  
  default:
    m_session->configuration().setMessageLevel( cond::Error );
  }
  m_session->configuration().setMessageLevel(cond::Debug);
  m_session->configuration().setAuthenticationMethod(cond::XML);
  m_session->configuration().setAuthenticationPath(authPath);
}
lumi::LumiOMDSReader::~LumiOMDSReader(){
  delete m_session;
}

void 
lumi::LumiOMDSReader::fill(int startRun,
			   int numberOfRuns,
			   std::vector< std::pair<lumi::LuminosityInfo*,cond::Time_t> >& result, short lumiVersionNumber){
  //fill summary info
  //fill detail info
  //select summary.DEADTIME_NORMALIZATION,summary.INSTANT_LUMI,summary.INSTANT_LUMI_ERR,summary.INSTANT_LUMI_QLTY,sect.LUMI_SECTION_NUMBER, bx.BUNCH_X_NUMBER,bx.ET_LUMI,bx.ET_LUMI_ERR,bx.ET_LUMI_QLTY from CMS_LUMI.LUMI_SUMMARIES summary, CMS_LUMI.LUMI_DETAILS bx, CMS_LUMI.LUMI_SECTIONS sect WHERE sect.SECTION_ID=summary.SECTION_ID AND sect.SECTION_ID=bx.SECTION_ID AND sect.lumi_section_number>0 AND sect.RUN_NUMBER=70674 and summary.lumi_version=1 ORDER BY sect.LUMI_SECTION_NUMBER
  
  try{
    m_session->open();
    cond::Connection con(m_constr,-1);
    con.connect(m_session);
    cond::CoralTransaction& transaction=con.coralTransaction();
    coral::AttributeList bindVariableList;
    bindVariableList.extend("runnumber",typeid(int));
    bindVariableList.extend("lumiversion",typeid(short));
    bindVariableList.extend("lsmin",typeid(int));
    bindVariableList["lumiversion"].data<short>()=lumiVersionNumber;
    bindVariableList["lsmin"].data<int>()=0;
    std::cout<<"lumiVersionNumber "<<lumiVersionNumber<<std::endl;
    transaction.start(true); 
    int stopRun=startRun+numberOfRuns;
    for( int currentRun=startRun;currentRun<stopRun;++currentRun){
      bindVariableList["runnumber"].data<int>()=currentRun;
      coral::IQuery* query1= transaction.nominalSchema().newQuery();
      query1->addToOutputList("sect.LUMI_SECTION_NUMBER","lumisectionid");
      query1->addToOutputList("summary.DEADTIME_NORMALIZATION","deadtime_norm");
      //query1->addToOutputList("summary.LUMI_VERSION","lumi_version");
      query1->addToOutputList("summary.INSTANT_LUMI","instant_lumi");
      query1->addToOutputList("summary.INSTANT_LUMI_ERR","instant_lumi_err");
      query1->addToOutputList("summary.INSTANT_LUMI_QLTY","instant_lumi_quality");      
      query1->addToOutputList("bx.BUNCH_X_NUMBER","bxidx");
      query1->addToOutputList("bx.ET_LUMI","bx_lumi_et");
      query1->addToOutputList("bx.ET_LUMI_ERR","bx_err_et");
      query1->addToOutputList("bx.ET_LUMI_QLTY","bx_quality_et");

      query1->addToOutputList("bx.OCC_LUMI_D1","bx_lumi_occd1");
      query1->addToOutputList("bx.OCC_LUMI_D1_ERR","bx_err_occd1");
      query1->addToOutputList("bx.OCC_LUMI_D1_QLTY","bx_quality_occd1");

      //query1->addToOutputList("bx.OCC_LUMI_D2","bx_lumi_occd2");
      //query1->addToOutputList("bx.OCC_LUMI_D2_ERR","bx_err_occd2");
      //query1->addToOutputList("bx.OCC_LUMI_D2_QLTY","bx_quality_occd2");

      query1->addToTableList( "LUMI_SUMMARIES","summary");
      query1->addToTableList( "LUMI_SECTIONS","sect");
      query1->addToTableList( "LUMI_DETAILS","bx" );

      query1->setCondition( "sect.SECTION_ID=summary.SECTION_ID AND sect.SECTION_ID=bx.SECTION_ID AND sect.RUN_NUMBER =:runnumber AND summary.LUMI_VERSION =:lumiversion AND sect.lumi_section_number > :lsmin", bindVariableList );
      query1->addToOrderList( "sect.LUMI_SECTION_NUMBER" );
      query1->addToOrderList( "bx.BUNCH_X_NUMBER" );
      query1->setRowCacheSize( 10692 );
      coral::ICursor& cursor1 = query1->execute();
      lumi::LuminosityInfo* l=0;
      int lastLumiSection=0;
      std::vector<lumi::BunchCrossingInfo> bxinfo_et;
      bxinfo_et.reserve(3564);
      std::vector<lumi::BunchCrossingInfo> bxinfo_occd1;
      bxinfo_occd1.reserve(3564);
      //std::vector<lumi::BunchCrossingInfo> bxinfo_occd2;
      //bxinfo_occd2.reserve(3564);

      if( !cursor1.next() ){
	///if run doesn't exist
	std::cout<<"run "<<currentRun<<" doesn't exist, do nothing"<<std::endl;
	//l=new lumi::LuminosityInfo;
	//edm::LuminosityBlockID lu(currentRun,1);
	//cond::Time_t current=(cond::Time_t)(lu.value());
	//l->setLumiNull();
	//l->setLumiSectionId(1);
	//result.push_back(std::make_pair<lumi::LuminosityInfo*,cond::Time_t>(l,current));
	continue;
      }else{
	while( cursor1.next() ){
	  const coral::AttributeList& row=cursor1.currentRow();
	  //row.toOutputStream(std::cout)<<std::endl;
	  int currentLumiSection=row["lumisectionid"].data<int>();//not null
	  //short lumiversion=row["lumi_version"].data<short>();//is part of the query condition, so cannot be null
	  int bxidx=(int)row["bxidx"].data<short>(); //not null
	  
	  std::cout<<"currentLumiSection "<<currentLumiSection<<std::endl;
	  std::cout<<"bxidx "<<bxidx<<std::endl;
	  
	  float bx_lumi_et=-99.0;
	  if( !row["bx_lumi_et"].isNull() ){
	    bx_lumi_et=(float)row["bx_lumi_et"].data<float>(); //if null, convert to negative
	  }
	  float bx_err_et=-99.0;
	  if( !row["bx_err_et"].isNull() ){
	    bx_err_et=(float)row["bx_err_et"].data<double>();
	  }
	  int bx_quality_et=-99;
	  if( !row["bx_quality_et"].isNull() ){
	    bx_quality_et=(int)row["bx_quality_et"].data<short>();
	  }
	  bxinfo_et.push_back(lumi::BunchCrossingInfo(bxidx,bx_lumi_et,bx_err_et,bx_quality_et));
	  
	  float bx_lumi_occd1=-99.0;
	  if( !row["bx_lumi_occd1"].isNull() ){
	    bx_lumi_occd1=(float)row["bx_lumi_occd1"].data<float>();
	  }
	  float bx_err_occd1=-99.0;
	  if( !row["bx_err_occd1"].isNull() ){
	    bx_err_occd1=(float)row["bx_err_occd1"].data<double>();
	  }	  
	  int bx_quality_occd1=-99;
	  if( !row["bx_quality_occd1"].isNull() ){
	    bx_quality_occd1=(int)row["bx_quality_occd1"].data<short>();
	  }
	  bxinfo_occd1.push_back(lumi::BunchCrossingInfo(bxidx,bx_lumi_occd1,bx_err_occd1,bx_quality_occd1));
	  /*  
	      float bx_lumi_occd2=-99.0;
	      if( !row["bx_lumi_occd2"].isNull() ){
	      bx_lumi_occd2=(float)row["bx_lumi_occd2"].data<double>();
	      }
	      float bx_err_occd2=-99.0;
	      if( !row["bx_err_occd2"].isNull() ){
	      bx_err_occd2=(float)row["bx_err_occd2"].data<double>();
	      }
	      int bx_quality_occd2=-99;
	      if( !row["bx_quality_occd2"].isNull() ){
	      bx_quality_occd2=(int)row["bx_quality_occd2"].data<short>();
	      }
	      bxinfo_occd2.push_back(lumi::BunchCrossingInfo(bxidx,bx_lumi_occd2,bx_err_occd2,bx_quality_occd2));
	  */
	  if(currentLumiSection>lastLumiSection && bxidx==3564){
	    l=new lumi::LuminosityInfo;
	    edm::LuminosityBlockID lu(currentRun,currentLumiSection);
	    cond::Time_t current=(cond::Time_t)(lu.value());
	    l->setLumiVersionNumber(lumiVersionNumber);
	    l->setLumiSectionId(currentLumiSection);
	    float deadfrac=-99.0;
	    if( !row["deadtime_norm"].isNull() ){
	      deadfrac=row["deadtime_norm"].data<float>();
	    }

	    float instant_lumi=-99.0;
	    if( !row["instant_lumi"].isNull() ){
	      instant_lumi=(float)row["instant_lumi"].data<float>();
	    }
	    float instant_lumi_err=-99.0;
	    if( !row["instant_lumi_err"].isNull() ){
	      instant_lumi_err=(float)row["instant_lumi_err"].data<double>();
	    }
	    int instant_lumi_quality=-99;
	    if( !row["instant_lumi_quality"].isNull() ){
	      instant_lumi_quality=(int)row["instant_lumi_quality"].data<short>();
	    }

	    l->setLumiAverage(instant_lumi);
	    l->setLumiError(instant_lumi_err);
	    l->setLumiQuality(instant_lumi_quality);
	    l->setDeadFraction(deadfrac);
	    l->setBunchCrossingData(bxinfo_et,lumi::ET);
	    l->setBunchCrossingData(bxinfo_occd1,lumi::OCCD1);
	    //l->setBunchCrossingData(bxinfo_occd2,lumi::OCCD2);
	    result.push_back(std::make_pair<lumi::LuminosityInfo*,cond::Time_t>(l,current));
	    bxinfo_et.clear();
	    bxinfo_occd1.clear();
	    //bxinfo_occd2.clear();
	    ++lastLumiSection;
	  }
	}
      }
      cursor1.close();
      delete query1;
      
      //std::cout<<6<<std::endl;
      //fill bx registry
      //select bx.BUNCH_X_NUMBER,bx.NORMALIZATION_ET,bx.ET_LUMI,bx.ET_LUMI_ERR,bx.ET_LUMI_QLTY,bx.NORMALIZATION_OCC_D1,bx.OCC_LUMI_D1,bx.OCC_LUMI_D1_ERR,bx.OCC_LUMI_D1_QLTY,bx.NORMALIZATION_OCC_D2,bx.OCC_LUMI_D2,bx.OCC_LUMI_D2_ERR,bx.OCC_LUMI_D2_QLTY from CMS_LUMI.LUMI_DETAILS bx, CMS_LUMI.LUMI_SECTIONS sect WHERE sect.SECTION_ID=bx.SECTION_ID AND sect.RUN_NUMBER=70674 ORDERBY sect.LUMI_SECTION_NUMBER;
    /*
      coral::IQuery* query2=transaction.nominalSchema().newQuery();
      query2->addToOutputList("bx.BUNCH_X_NUMBER");
      query2->addToOutputList("bx.NORMALIZATION_ET");
      query2->addToOutputList("bx.ET_LUMI" );
      query2->addToOutputList("bx.ET_LUMI_ERR" );
      query2->addToOutputList("bx.ET_LUMI_QLTY" );
      query2->addToOutputList("bx.NORMALIZATION_OCC_D1" );
      query2->addToOutputList("bx.OCC_LUMI_D1" );
      query2->addToOutputList("bx.OCC_LUMI_D1_ERR");
      query2->addToOutputList("bx.OCC_LUMI_D1_QLTY");
      query2->addToOutputList("bx.NORMALIZATION_OCC_D2");
      query2->addToOutputList("bx.OCC_LUMI_D2");
      query2->addToOutputList("bx.OCC_LUMI_D2_ERR");
      query2->addToOutputList("bx.OCC_LUMI_D2_QLTY";
      query2->addToOutputList("sect.LUMI_SECTION_NUMBER");
      query2->addToOutputList("sect.RUN_NUMBER");
      query2->addToTableList( "LUMI_DETAILS","bx");
      query2->addToTableList( "LUMI_SECTIONS","sect");
      query2->setCondition( "sect.SECTION_ID=bx.SECTION_ID AND sect.RUN_NUMBER BETWEEN :startrun AND :stoprun", bindVariableList );
      query2->addToOrderList( "sect.LUMI_SECTION_NUMBER" );
      query2->setRowCacheSize( 10000 );
      coral::ICursor& cursor2 = query2->execute();
      while( cursor2.next() ){
      const coral::AttributeList& row=cursor2.currentRow();
      row.toOutputStream(std::cout)<<std::endl;
      }
      std::cout<<7<<std::endl;
      cursor2.close();
      delete query2;
    */
    }
    transaction.commit(); 
    std::cout<<"committed"<<std::endl;
  }catch(const std::exception& er){
    std::cout<<"caught exception "<<er.what()<<std::endl;
  }
}

DEFINE_EDM_PLUGIN(lumi::LumiReaderFactory,lumi::LumiOMDSReader,"omds");

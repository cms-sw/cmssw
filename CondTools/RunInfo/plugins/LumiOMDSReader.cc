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
void lumi::LumiOMDSReader::fill(int startRun,
				int numberOfRuns,
				std::vector< std::pair<lumi::LuminosityInfo*,cond::Time_t> >& result){
  //fill hlt registry hlt is empty. nothing to do
  //select hlt.INPUT_COUNT, hlt.ACCEPT_COUNT, hlt.PRESCALE_FACTOR from CMS_LUMI.HLTS hlt , CMS_LUMI.LUMI_SECTIONS sect WHERE sect.SECTION_ID=hlt.SECTION_ID AND sect.RUN_NUMBER=runnumber AND sect.LUMI_SECTION_NUMBER=lumisectionid;

  //fill lumisummary registry
  //select summary.DEADTIME_NORMALIZATION, summary.NORMALIZATION,summary.INSTANT_LUMI,summary.INSTANT_LUMI_ERR,summary.INSTANT_LUMI_QLTY, summary.NORMALIZATION_ET,summary.INSTANT_ET_LUMI,summary.INSTANT_ET_LUMI_ERR,summary.INSTANT_ET_LUMI_QLTY,summary.NORMALIZATION_OCC_D1,summary.INSTANT_OCC_LUMI_D1,summary.INSTANT_OCC_LUMI_D1_ERR,summary.INSTANT_OCC_LUMI_D1_QLTY,summary.NORMALIZATION_OCC_D2,summary.INSTANT_OCC_LUMI_D2_ERR, summary.INSTANT_OCC_LUMI_D2_QLTY, sect.LUMI_SECTION_NUMBER from CMS_LUMI.LUMI_SUMMARIES summary, CMS_LUMI.LUMI_SECTIONS sect WHERE sect.SECTION_ID=summary.SECTION_ID AND sect.RUN_NUMBER=70674 ORDER BY sect.LUMI_SECTION_NUMBER ;
  //select summary.DEADTIME_NORMALIZATION, summary.NORMALIZATION,summary.INSTANT_LUMI,summary.INSTANT_LUMI_ERR,summary.INSTANT_LUMI_QLTY, summary.NORMALIZATION_ET,summary.INSTANT_ET_LUMI,summary.INSTANT_ET_LUMI_ERR,summary.INSTANT_ET_LUMI_QLTY,summary.NORMALIZATION_OCC_D1,summary.INSTANT_OCC_LUMI_D1,summary.INSTANT_OCC_LUMI_D1_ERR,summary.INSTANT_OCC_LUMI_D1_QLTY,summary.NORMALIZATION_OCC_D2,summary.INSTANT_OCC_LUMI_D2_ERR, summary.INSTANT_OCC_LUMI_D2_QLTY, sect.LUMI_SECTION_NUMBER, bx.BUNCH_X_NUMBER,bx.NORMALIZATION_ET,bx.ET_LUMI,bx.ET_LUMI_ERR,bx.ET_LUMI_QLTY,bx.NORMALIZATION_OCC_D1,bx.OCC_LUMI_D1,bx.OCC_LUMI_D1_ERR,bx.OCC_LUMI_D1_QLTY,bx.NORMALIZATION_OCC_D2,bx.OCC_LUMI_D2,bx.OCC_LUMI_D2_ERR,bx.OCC_LUMI_D2_QLTY from CMS_LUMI.LUMI_SUMMARIES summary, CMS_LUMI.LUMI_DETAILS bx, CMS_LUMI.LUMI_SECTIONS sect WHERE sect.SECTION_ID=summary.SECTION_ID AND sect.SECTION_ID=bx.SECTION_ID AND sect.RUN_NUMBER=70674 ORDER BY sect.LUMI_SECTION_NUMBER
  try{
    m_session->open();
    cond::Connection con(m_constr,-1);
    con.connect(m_session);
    cond::CoralTransaction& transaction=con.coralTransaction();
    coral::AttributeList bindVariableList;
    bindVariableList.extend("runnumber",typeid(int));
    transaction.start(true); 
    int stopRun=startRun+numberOfRuns;
    for( int currentRun=startRun;currentRun<stopRun;++currentRun){
      bindVariableList["runnumber"].data<int>()=currentRun;
      coral::IQuery* query1= transaction.nominalSchema().newQuery();
      query1->addToOutputList("sect.LUMI_SECTION_NUMBER","lumisectionid");
      query1->addToOutputList("summary.DEADTIME_NORMALIZATION","deadtime_norm");
      /*comment out until meaning is clear
      query1->addToOutputList("summary.NORMALIZATION","norm");
      query1->addToOutputList("summary.INSTANT_LUMI","value");
      query1->addToOutputList("summary.INSTANT_LUMI_ERR","err");
      query1->addToOutputList("summary.INSTANT_LUMI_QLTY","quality");
      */
      query1->addToOutputList("summary.NORMALIZATION_ET","norm_et");
      query1->addToOutputList("summary.INSTANT_ET_LUMI","value_et");
      query1->addToOutputList("summary.INSTANT_ET_LUMI_ERR","err_et");
      query1->addToOutputList("summary.INSTANT_ET_LUMI_QLTY","quality_et");
      
      query1->addToOutputList("summary.NORMALIZATION_OCC_D1","norm_occd1");
      query1->addToOutputList("summary.INSTANT_OCC_LUMI_D1","value_occd1");
      query1->addToOutputList("summary.INSTANT_OCC_LUMI_D1_ERR","err_occd1");
      query1->addToOutputList("summary.INSTANT_OCC_LUMI_D1_QLTY","quality_occd1");
      query1->addToOutputList("summary.NORMALIZATION_OCC_D2","norm_occd2");
      query1->addToOutputList("summary.INSTANT_OCC_LUMI_D2","value_occd2");
      query1->addToOutputList("summary.INSTANT_OCC_LUMI_D2_ERR","err_occd2");
      query1->addToOutputList("summary.INSTANT_OCC_LUMI_D2_QLTY","quality_occd2");
      
      query1->addToOutputList("bx.BUNCH_X_NUMBER","bxidx");

      query1->addToOutputList("bx.NORMALIZATION_ET","bx_norm_et");
      query1->addToOutputList("bx.ET_LUMI","bx_lumi_et");
      query1->addToOutputList("bx.ET_LUMI_ERR","bx_err_et");
      query1->addToOutputList("bx.ET_LUMI_QLTY","bx_quality_et");

      query1->addToOutputList("bx.NORMALIZATION_OCC_D1","bx_norm_occd1");
      query1->addToOutputList("bx.OCC_LUMI_D1","bx_lumi_occd1");
      query1->addToOutputList("bx.OCC_LUMI_D1_ERR","bx_err_occd1");
      query1->addToOutputList("bx.OCC_LUMI_D1_QLTY","bx_quality_occd1");

      query1->addToOutputList("bx.NORMALIZATION_OCC_D2","bx_norm_occd2");
      query1->addToOutputList("bx.OCC_LUMI_D2","bx_lumi_occd2");
      query1->addToOutputList("bx.OCC_LUMI_D2_ERR","bx_err_occd2");
      query1->addToOutputList("bx.OCC_LUMI_D2_QLTY","bx_quality_occd2");

      query1->addToTableList( "LUMI_SUMMARIES","summary");
      query1->addToTableList( "LUMI_SECTIONS","sect");
      query1->addToTableList( "LUMI_DETAILS","bx" );

      query1->setCondition( "sect.SECTION_ID=summary.SECTION_ID AND sect.SECTION_ID=bx.SECTION_ID AND sect.RUN_NUMBER =:runnumber", bindVariableList );
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
      std::vector<lumi::BunchCrossingInfo> bxinfo_occd2;
      bxinfo_occd2.reserve(3564);
      while( cursor1.next() ){
	const coral::AttributeList& row=cursor1.currentRow();
	//row.toOutputStream(std::cout)<<std::endl;
	int currentLumiSection=row["lumisectionid"].data<int>();
	int bxidx=(int)row["bxidx"].data<short>();
	//std::cout<<"currentLumiSection "<<currentLumiSection<<std::endl;
	//std::cout<<"bxidx "<<bxidx<<std::endl;
	float bx_lumi_et=(float)row["bx_lumi_et"].data<double>();
	float bx_err_et=(float)row["bx_err_et"].data<double>();
	int bx_quality_et=(int)row["bx_quality_et"].data<short>();
	int bx_norm_et=(int)row["bx_norm_et"].data<long long>();
		
	bxinfo_et.push_back(lumi::BunchCrossingInfo(bxidx,bx_lumi_et,bx_err_et,bx_quality_et,bx_norm_et));
	
	float bx_lumi_occd1=(float)row["bx_lumi_occd1"].data<double>();
	float bx_err_occd1=(float)row["bx_err_occd1"].data<double>();
	int bx_quality_occd1=(int)row["bx_quality_occd1"].data<short>();
	int bx_norm_occd1=(int)row["bx_norm_occd1"].data<long long>();
	bxinfo_occd1.push_back(lumi::BunchCrossingInfo(bxidx,bx_lumi_occd1,bx_err_occd1,bx_quality_occd1,bx_norm_occd1));

	float bx_lumi_occd2=(float)row["bx_lumi_occd2"].data<double>();
	float bx_err_occd2=(float)row["bx_err_occd2"].data<double>();
	int bx_quality_occd2=(int)row["bx_quality_occd2"].data<short>();
	int bx_norm_occd2=(int)row["bx_norm_occd2"].data<long long>();
	bxinfo_occd2.push_back(lumi::BunchCrossingInfo(bxidx,bx_lumi_occd2,bx_err_occd2,bx_quality_occd2,bx_norm_occd2));
	if(currentLumiSection>lastLumiSection && bxidx>=3564){
	  l=new lumi::LuminosityInfo;
	  edm::LuminosityBlockID lu(currentRun,currentLumiSection);
	  cond::Time_t current=(cond::Time_t)(lu.value());
	  l->setLumiSectionId(currentLumiSection);
	  float value_et=(float)row["value_et"].data<double>();
	  float err_et=(float)row["err_et"].data<double>();
	  int quality_et=(int)row["quality_et"].data<short>();
	  int norm_et=(int)row["norm_et"].data<long long>();
	  lumi::LumiAverage avg_et(value_et,err_et,quality_et,norm_et);
	  l->setLumiAverage(avg_et,lumi::ET);

	  float value_occd1=(float)row["value_occd1"].data<double>();
	  float err_occd1=(float)row["err_occd1"].data<double>();
	  int quality_occd1=(int)row["quality_occd1"].data<short>();
	  int norm_occd1=(int)row["norm_occd1"].data<long long>();
	  lumi::LumiAverage avg_occd1(value_occd1,err_occd1,quality_occd1,norm_occd1);
	  l->setLumiAverage(avg_occd1,lumi::OCCD1);

	  float value_occd2=(float)row["value_occd2"].data<double>();
	  float err_occd2=(float)row["err_occd2"].data<double>();
	  int quality_occd2=(int)row["quality_occd2"].data<short>();
	  int norm_occd2=(int)row["norm_occd2"].data<long long>();
	  lumi::LumiAverage avg_occd2(value_occd2,err_occd2,quality_occd2,norm_occd2);
	  l->setLumiAverage(avg_occd2,lumi::OCCD2);
	  l->setBunchCrossingData(bxinfo_et,lumi::ET);
	  l->setBunchCrossingData(bxinfo_occd1,lumi::OCCD1);
	  l->setBunchCrossingData(bxinfo_occd2,lumi::OCCD2);
	  result.push_back(std::make_pair<lumi::LuminosityInfo*,cond::Time_t>(l,current));
	  bxinfo_et.clear();
	  bxinfo_occd1.clear();
	  bxinfo_occd2.clear();
	  ++lastLumiSection;
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

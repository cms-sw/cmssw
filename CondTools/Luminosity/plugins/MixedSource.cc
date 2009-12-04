#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "CondCore/DBCommon/interface/Time.h"
#include "CondFormats/Luminosity/interface/LumiSectionData.h"
#include "CondTools/Luminosity/interface/LumiRetrieverFactory.h"
#include "RelationalAccess/IAuthenticationService.h"
#include "RelationalAccess/IConnectionService.h"
#include "RelationalAccess/IConnectionServiceConfiguration.h"
#include "CoralKernel/Context.h"
#include "CoralKernel/IHandle.h"
#include "CoralKernel/IProperty.h"
#include "CoralKernel/IPropertyManager.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/AttributeSpecification.h"
#include "CoralBase/Exception.h"
#include "CoralBase/TimeStamp.h"
#include "CoralBase/MessageStream.h"
#include "RelationalAccess/AccessMode.h"
#include "RelationalAccess/ConnectionService.h"
#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/IConnectionServiceConfiguration.h"
#include "RelationalAccess/ITransaction.h"
#include "RelationalAccess/ITypeConverter.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ICursor.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/IView.h"
#include "RelationalAccess/ITable.h"
#include "LumiDataStructures.h"
#include "MixedSource.h"
#include <memory>
#include <iostream>
#include <boost/filesystem/operations.hpp>
#include <boost/regex.hpp>
#include "TFile.h"
#include "TTree.h"

lumi::MixedSource::MixedSource(const edm::ParameterSet& pset):LumiRetrieverBase(pset),m_filename(""),m_lumiversion("-1"){
  m_mode=pset.getUntrackedParameter<std::string>("runmode","truerun");
  std::cout<<"mode "<<m_mode<<std::endl;
  if(m_mode!="trgdryrun"){
    if(!pset.exists("lumiFileName")){
      throw std::runtime_error(std::string("parameter lumiFileName is required for mode ")+m_mode);
    }
    m_filename=pset.getParameter<std::string>("lumiFileName");
    std::cout<<"m_filename "<<m_filename<<std::endl;
    boost::regex re("_");
    boost::sregex_token_iterator p(m_filename.begin(),m_filename.end(),re,-1);
    boost::sregex_token_iterator end;
    std::vector<std::string> vecstrResult;
    while(p!=end){
      vecstrResult.push_back(*p++);
    }
    std::string runstr=*(vecstrResult.end()-3);
    m_run=this->str2int(runstr);
    std::string::size_type idx,pos;
    idx=m_filename.rfind("_");
    pos=m_filename.rfind(".");
    m_lumiversion=m_filename.substr(idx+1,pos-idx-1);
    //std::cout<<"runnumber "<<m_run<<std::endl;
    //std::cout<<"lumi version "<<m_lumiversion<<std::endl;
    m_source=TFile::Open(m_filename.c_str(),"READ");
  }else{
    m_run=pset.getUntrackedParameter<unsigned int>("runnumber",1);
  }
  if(m_mode!="lumidryrun"){
    if(!pset.exists("triggerDB")||!pset.exists("authPath")){
      throw std::runtime_error(std::string("parameter triggerDB and authPath are required for mode ")+m_mode);
    }
    m_trgdb=pset.getParameter<std::string>("triggerDB");
    m_authpath=pset.getParameter<std::string>("authPath");
  }
}
void 
lumi::MixedSource::initDB() {
  std::string authfile("authentication.xml");
  boost::filesystem::path authPath(m_authpath);
  if(boost::filesystem::is_directory(m_authpath)){
    authPath /= boost::filesystem::path(authfile);
  }
  std::string authName=authPath.string();
  coral::Context::instance().PropertyManager().property("AuthenticationFile")->set(authName);
   m_dbservice = new coral::ConnectionService();
   m_dbservice->configuration().setAuthenticationService("CORAL/Services/XMLAuthenticationService");
   m_dbservice->configuration().disablePoolAutomaticCleanUp();
   m_dbservice->configuration().setConnectionTimeOut(0);
   coral::MessageStream::setMsgVerbosity(coral::Error);
}
void
lumi::MixedSource::getLumiData(const std::string& filename,
			       lumi::MixedSource::LumiResult& lumiresult
			       ){ 
  unsigned int runnumber=0;
  TTree *hlxtree = (TTree*)m_source->Get("HLXData");
  if(!hlxtree){
    throw std::runtime_error(std::string("non-existing HLXData "));
  }
  //hlxtree->Print();
  std::auto_ptr<HCAL_HLX::LUMI_SECTION> localSection(new HCAL_HLX::LUMI_SECTION);
  HCAL_HLX::LUMI_SECTION_HEADER* lumiheader = &(localSection->hdr);
  HCAL_HLX::LUMI_SUMMARY* lumisummary = &(localSection->lumiSummary);
  HCAL_HLX::LUMI_DETAIL* lumidetail = &(localSection->lumiDetail);
  
  hlxtree->SetBranchAddress("Header.",&lumiheader);
  hlxtree->SetBranchAddress("Summary.",&lumisummary);
  hlxtree->SetBranchAddress("Detail.",&lumidetail);

  size_t nentries=hlxtree->GetEntries();
  size_t ncmslumi=0;
  //std::cout<<"processing total lumi lumisection "<<nentries<<std::endl;
  //size_t lumisecid=0;
  //unsigned int lumilumisecid=0;
  runnumber=lumiheader->runNumber;
  for(size_t i=0;i<nentries;++i){
    hlxtree->GetEntry(i);
    if(!lumiheader->bCMSLive){
      std::cout<<"non-CMS LS "<<lumiheader->sectionNumber<<std::endl;
      continue;
    }else{
      ++ncmslumi;
    }
    lumi::MixedSource::PerLumiData h;
    h.bxET.reserve(3564);
    h.bxOCC1.reserve(3564);
    h.bxOCC2.reserve(3564);

    runnumber=lumiheader->runNumber;
    if(runnumber!=m_run) throw std::runtime_error(std::string("requested run ")+this->int2str(m_run)+" does not match runnumber in the data header "+this->int2str(runnumber));
    h.lumilsnr=lumiheader->sectionNumber;
    h.cmslsnr=ncmslumi;//we record cms lumils
    h.startorbit=lumiheader->startOrbit;
    h.lumiavg=lumisummary->InstantLumi;
    
    for(size_t i=0;i<3564;++i){
      lumi::MixedSource::PerBXData bET;
      lumi::MixedSource::PerBXData bOCC1;
      lumi::MixedSource::PerBXData bOCC2;
      bET.idx=i+1;
      bET.lumivalue=lumidetail->ETLumi[i];
      bET.lumierr=lumidetail->ETLumiErr[i];
      bET.lumiquality=lumidetail->ETLumiQlty[i];      
      //bxinfoET.push_back(lumi::BunchCrossingInfo(i+1,lumidetail->ETLumi[i],lumidetail->ETLumiErr[i],lumidetail->ETLumiQlty[i]));
      h.bxET.push_back(bET);

      bOCC1.idx=i+1;
      bOCC1.lumivalue=lumidetail->OccLumi[0][i];
      bOCC1.lumierr=lumidetail->OccLumiErr[0][i];
      bOCC1.lumiquality=lumidetail->OccLumiQlty[0][i]; 
      h.bxOCC1.push_back(bOCC1);
      //bxinfoOCC1.push_back(lumi::BunchCrossingInfo(i+1,lumidetail->OccLumi[0][i],lumidetail->OccLumiErr[0][i],lumidetail->OccLumiQlty[0][i]));
      
      bOCC2.idx=i+1;
      bOCC2.lumivalue=lumidetail->OccLumi[1][i];
      bOCC2.lumierr=lumidetail->OccLumiErr[1][i];
      bOCC2.lumiquality=lumidetail->OccLumiQlty[1][i]; 
      h.bxOCC2.push_back(bOCC2);
      //bxinfoOCC2.push_back(lumi::BunchCrossingInfo(i+1,lumidetail->OccLumi[1][i],lumidetail->OccLumiErr[1][i],lumidetail->OccLumiQlty[1][i]));
      
    }
    lumiresult.push_back(h);
  }
}
void
lumi::MixedSource::getTrgData(unsigned int runnumber,
			    coral::ISessionProxy* session,
			    lumi::MixedSource::TriggerNameResult_Algo& algonames,
			    lumi::MixedSource::TriggerNameResult_Tech& technames,
			    lumi::MixedSource::PrescaleResult_Algo& algoprescale,
			    lumi::MixedSource::PrescaleResult_Tech& techprescale,
			    lumi::MixedSource::TriggerCountResult_Algo& algocount,
			    lumi::MixedSource::TriggerCountResult_Tech& techcount,
			    lumi::MixedSource::TriggerDeadCountResult& deadtime
			      ){
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
    throw std::runtime_error(std::string("non-existing view ")+algoviewname);
  }
  if(!gtmonschemaHandle.existsView(techviewname)){
    throw std::runtime_error(std::string("non-existing view ")+techviewname);
  }
  if(!gtmonschemaHandle.existsView(deadviewname)){
    throw std::runtime_error(std::string("non-existing view ")+deadviewname);
  }
  if(!gtmonschemaHandle.existsTable(celltablename)){
    throw std::runtime_error(std::string("non-existing table ")+celltablename);
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
  lumi::MixedSource::BITCOUNT mybitcount_algo;
  mybitcount_algo.reserve(128);
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
    std::cout<<"requested run "<<runnumber<<" doesn't exist for algocounts"<<std::endl;
    c.close();
    delete Queryalgoview;
    transaction.commit();
    return;
  }
  delete Queryalgoview;
  //
  //select counts,lsnr,techbit from cms_gt_mon.gt_mon_trig_tech_view where runnr=:runnumber order by lsnr,techbit;
  //
  lumi::MixedSource::BITCOUNT mybitcount_tech; 
  mybitcount_tech.reserve(64);
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
    std::cout<<"requested run "<<runnumber<<" doesn't exist for techcounts, do nothing"<<std::endl;
    techcursor.close();
    delete Querytechview;
    transaction.commit();
    return;
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
    deadtime.push_back(count);
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
  //printCountResult(countresult_algo,countresult_tech);
  //printDeadTimeResult(deadtimeresult);
  
  /**
     Part II
     query tables in schema cms_gt
  **/
  //transaction.start(true);
  coral::ISchema& gtschemaHandle=session->schema(gtschema);
  if(!gtschemaHandle.existsView(runtechviewname)){
    throw std::runtime_error(std::string("non-existing view ")+runtechviewname);
  }
  if(!gtschemaHandle.existsView(runalgoviewname)){
    throw std::runtime_error(std::string("non-existing view ")+runalgoviewname);
  }
  if(!gtschemaHandle.existsView(runprescalgoviewname)){
    throw std::runtime_error(std::string("non-existing view ")+runprescalgoviewname);
  }
  if(!gtschemaHandle.existsView(runpresctechviewname)){
    throw std::runtime_error(std::string("non-existing view ")+runpresctechviewname);
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
  for(int bitidx=0;bitidx<128;++bitidx){
    std::string algopresc=algoprescBase+int2str(bitidx);
    qAlgoPrescOutput.extend(algopresc,typeid(unsigned int));
  }
  for(int bitidx=0;bitidx<128;++bitidx){
    std::string algopresc=algoprescBase+int2str(bitidx);
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
    for(int bitidx=0;bitidx<128;++bitidx){
      std::string algopresc=algoprescBase+int2str(bitidx);
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
  for(int bitidx=0;bitidx<64;++bitidx){
    std::string techpresc=techprescBase+this->int2str(bitidx);
    qTechPrescOutput.extend(techpresc,typeid(unsigned int));
  }
  for(int bitidx=0;bitidx<64;++bitidx){
    std::string techpresc=techprescBase+int2str(bitidx);
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
    for(int bitidx=0;bitidx<64;++bitidx){
      std::string techpresc=techprescBase+int2str(bitidx);
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
}
std::string 
lumi::MixedSource::int2str(int t){
  std::stringstream ss;
  ss.width(3);
  ss.fill('0');
  ss<<t;
  return ss.str();
}
unsigned int
lumi::MixedSource::str2int(const std::string& s){
  std::istringstream myStream(s);
  unsigned int i;
  if(myStream>>i){
    return i;
  }else{
    throw std::runtime_error(std::string("str2int error"));
  }
}
void 
lumi::MixedSource::printCountResult(
		const lumi::MixedSource::TriggerCountResult_Algo& algo,
		const lumi::MixedSource::TriggerCountResult_Tech& tech){
  size_t lumisec=1;
  std::cout<<"===Algorithm trigger counts==="<<algo.size()<<std::endl;
  std::vector<unsigned long long> totalalgocounts(128);
  std::vector<unsigned long long> totaltechcounts(64);
  std::fill(totaltechcounts.begin(),totaltechcounts.end(),0);
  for(lumi::MixedSource::TriggerCountResult_Algo::const_iterator it=algo.begin();it!=algo.end();++it){
    std::cout<<"lumisec "<<lumisec<<std::endl;
    ++lumisec;
    //std::cout<<"total bits "<<it->size()<<std::endl;
    size_t bitidx=0;
    for(lumi::MixedSource::BITCOUNT::const_iterator itt=it->begin();itt!=it->end();++itt){
      std::cout<<"\t bit: "<<bitidx<<" : count : "<<*itt<<std::endl;
      //std::cout<<"count before "<<totalalgocounts.at(bitidx)<<std::endl;
      totalalgocounts.at(bitidx)+=*itt;
      //std::cout<<"count after "<<totalalgocounts.at(bitidx)<<std::endl;
      if (it==algo.end()-1){ //last LS print sum
	std::cout<<"\t===Total counts by bit "<<bitidx<<" : "<<totalalgocounts.at(bitidx)<<std::endl;
      }
      ++bitidx;
    }
  }
  
  std::cout<<"===Technical trigger counts==="<<tech.size()<<std::endl;
  lumisec=1;//reset lumisec counter
  for(lumi::MixedSource::TriggerCountResult_Tech::const_iterator it=tech.begin();it!=tech.end();++it){
    std::cout<<"lumisec "<<lumisec<<std::endl;
    ++lumisec;
    size_t bitidx=0;
    for(BITCOUNT::const_iterator itt=it->begin();itt!=it->end();++itt){
      std::cout<<"\t bit: "<<bitidx<<" : count : "<<*itt<<std::endl;
      totaltechcounts[bitidx]+=*itt;
      if (it==tech.end()-1){ //last LS print sum
	std::cout<<"\t===Total counts by bit "<<bitidx<<" : "<<totaltechcounts.at(bitidx)<<std::endl;
      }
      ++bitidx;
    }
  }
  
}
void 
lumi::MixedSource::printDeadTimeResult(const lumi::MixedSource::TriggerDeadCountResult& result){
  size_t lumisec=1;
  unsigned long long totaldead=0;
  std::cout<<"===Deadtime counts==="<<result.size()<<std::endl;
  for(lumi::MixedSource::TriggerDeadCountResult::const_iterator it=result.begin();it!=result.end();++it){
    std::cout<<"lumisec "<<lumisec<<" : counts : "<<*it<<std::endl;
    totaldead += *it;
    ++lumisec;
  }
  std::cout<<"===Total Deadtime counts per run over "<<lumisec-1<<" LS "<<totaldead<<std::endl;
}
void 
lumi::MixedSource::printTriggerNameResult(
		const lumi::MixedSource::TriggerNameResult_Algo& algonames,
		const lumi::MixedSource::TriggerNameResult_Tech& technames){
  size_t bitidx=0;
  std::cout<<"===Algorithm trigger bit name==="<<std::endl;
  for(lumi::MixedSource::TriggerNameResult_Algo::const_iterator it=algonames.begin();it!=algonames.end();++it){
    std::cout<<"\t bit: "<<bitidx<<" : name : "<<*it<<std::endl;
    ++bitidx;    
  }
  bitidx=0;
  std::cout<<"===Tech trigger bit name==="<<std::endl;
  for(lumi::MixedSource::TriggerNameResult_Tech::const_iterator it=technames.begin();it!=technames.end();++it){
    std::cout<<"\t bit: "<<bitidx<<" : name : "<<*it<<std::endl;
    ++bitidx;    
  }
}
void 
lumi::MixedSource::printPrescaleResult(
		const lumi::MixedSource::PrescaleResult_Algo& algo,
		const lumi::MixedSource::PrescaleResult_Tech& tech){
  size_t bitidx=0;
  std::cout<<"===Algorithm trigger bit prescale==="<<std::endl;
  for(lumi::MixedSource::PrescaleResult_Algo::const_iterator it=algo.begin();
      it!=algo.end();++it){
    std::cout<<"\t bit: "<<bitidx<<" : prescale : "<<*it<<std::endl;
    ++bitidx;    
  }
  bitidx=0;
  std::cout<<"===Tech trigger bit prescale==="<<std::endl;
  for(lumi::MixedSource::PrescaleResult_Tech::const_iterator it=tech.begin();
      it!=tech.end();++it){
    std::cout<<"\t bit: "<<bitidx<<" : prescale : "<<*it<<std::endl;
    ++bitidx;    
  }
}
void 
lumi::MixedSource::printLumiResult(
		const lumi::MixedSource::LumiResult& lumiresult){
  std::cout<<"===Lumi mesurement==="<<lumiresult.size()<<std::endl;
  for(lumi::MixedSource::LumiResult::const_iterator it=lumiresult.begin();
      it!=lumiresult.end();++it){
    std::cout<<"\t lumisec: "<<it->cmslsnr<<" : lumilumisec : "<<it->lumilsnr<<" : avg : "<<it->lumiavg<<" : startorbit : "<<it->startorbit<<std::endl;
  }
}
void 
lumi::MixedSource::printIntegratedLumi(
		const lumi::MixedSource::LumiResult& lumiresult,
		const lumi::MixedSource::TriggerDeadCountResult& deadtime){
  std::cout<<"Integrated Lumi "<<lumiresult.size()<<" "<<deadtime.size()<<std::endl;
  float delivered=0.0;
  float recorded=0.0;
  for(unsigned int i=0;i<lumiresult.size();++i){
    delivered+=lumiresult.at(i).lumiavg*93.2;
    unsigned int deadcountPerLS=deadtime.at(i);
    float deadfraction=deadcountPerLS*25.0*(1.0e-9)/93.2;
    std::cout<<i+1<<" deadfraction "<<deadfraction<<std::endl;
    recorded+=lumiresult.at(i).lumiavg*93.2*(1.0-deadfraction);
  }
  std::cout<<"LHC Delivered : "<<delivered<<" : CMS recorded : "<<recorded<<std::endl;
}

const std::string
lumi::MixedSource::fill(std::vector< std::pair<lumi::LumiSectionData*,cond::Time_t> >& result , bool allowForceFirstSince ){
  lumi::MixedSource::TriggerNameResult_Algo algonames;
  algonames.reserve(128);
  lumi::MixedSource::TriggerNameResult_Tech technames;
  technames.reserve(64);
  lumi::MixedSource::PrescaleResult_Algo algoprescale;
  algoprescale.reserve(128);
  lumi::MixedSource::PrescaleResult_Tech techprescale;
  techprescale.reserve(64);
  lumi::MixedSource::TriggerCountResult_Algo algocount;
  algocount.reserve(1024);
  lumi::MixedSource::TriggerCountResult_Tech techcount;
  techcount.reserve(1024);
  lumi::MixedSource::TriggerDeadCountResult deadtime;
  deadtime.reserve(400);
  lumi::MixedSource::LumiResult lumiresult;
  lumiresult.reserve(400);
  if(m_mode=="trgdryrun"){
    //=====query trigger db=====
    this->initDB();
    coral::ISessionProxy* session=m_dbservice->connect(m_trgdb, coral::ReadOnly);
    try{
      this->getTrgData(m_run,session,algonames,technames,algoprescale,techprescale,algocount,techcount,deadtime);
    }catch(const coral::Exception& er){
      std::cout<<"database problem "<<er.what()<<std::endl;
      delete session;
    }
    delete session;
    this->printTriggerNameResult(algonames,technames);
    this->printPrescaleResult(algoprescale,techprescale);
    this->printCountResult(algocount,techcount);
    this->printDeadTimeResult(deadtime);
    //std::cout<<"total cms lumi "<<ncmslumi<<std::endl;
    return std::string("trgdryrun ")+m_trgdb;
  }else if(m_mode=="lumidryrun"){
    //lumi::MixedSource::LumiResult lumiresult;
    this->getLumiData(m_filename,lumiresult);
    this->printLumiResult(lumiresult);
    return std::string("lumidryrun ")+m_filename+";"+m_lumiversion;
  }else if( m_mode=="dryrun" || m_mode=="truerun" ){
    //take run number from filename; compare with runnumber field 
    //take lumi info from file
    //take trigger data from db
    std::cout<<"run "<<m_run<<std::endl;
    this->initDB();
    coral::ISessionProxy* session=m_dbservice->connect(m_trgdb, coral::ReadOnly);
    try{
      this->getTrgData(m_run,session,algonames,technames,algoprescale,techprescale,algocount,techcount,deadtime);
    }catch(const coral::Exception& er){
      std::cout<<"database problem "<<er.what()<<std::endl;
      delete session;
      throw er;
    }
    delete session;
    this->getLumiData(m_filename,lumiresult);
    if(m_mode=="dryrun"){
      this->printTriggerNameResult(algonames,technames);
      this->printPrescaleResult(algoprescale,techprescale);
      this->printCountResult(algocount,techcount);
      this->printDeadTimeResult(deadtime);
      this->printLumiResult(lumiresult);
      this->printIntegratedLumi(lumiresult,deadtime);
      return std::string("mixedsource dryrun ")+m_filename+";"+m_lumiversion;
    }else{
      //for the moment, lumi data drives the loop
      if(deadtime.size()!=lumiresult.size()){
	std::cout<<"WARNING: inconsistent number of trg and lumi lumisections"<<std::endl;
	//throw std::runtime_error(std::string("inconsistent number of lumisections"));
      }
      unsigned int runnumber;
      if(allowForceFirstSince){ //if allowForceFirstSince and this is the head of the iov, then set the head to the begin of time
	runnumber=1;
      }else{
	runnumber=m_run;
      }
      //this->printDeadTimeResult(deadtime);
      LumiResult::const_iterator lit;
      LumiResult::const_iterator litBeg=lumiresult.begin();
      LumiResult::const_iterator litEnd=lumiresult.end();
      unsigned long long totaldeadtime=0;
      for(lit=litBeg;lit!=litEnd;++lit){
	unsigned int lumisecid=lit->cmslsnr;//start from 1
	edm::LuminosityBlockID lu(runnumber,lumisecid);
	//should fill cms lumiid!
	std::cout<<"==== run lumiid lumiversion "<<runnumber<<"\t"<<lumisecid<<"\t"<<m_lumiversion<<std::endl;
	cond::Time_t current=(cond::Time_t)(lu.value());
	lumi::LumiSectionData* l=new lumi::LumiSectionData;
	l->setLumiVersion(m_lumiversion);
	l->setLumiSectionId(lumisecid);
	//std::cout<<"current "<<current<<std::endl;
	l->setStartOrbit((unsigned long long)lit->startorbit);
	l->setLumiAverage(lit->lumiavg);

	std::vector<lumi::BunchCrossingInfo> bxinfoET;
	std::vector<lumi::BunchCrossingInfo> bxinfoOCC1;
	std::vector<lumi::BunchCrossingInfo> bxinfoOCC2;
	bxinfoET.reserve(3564);
	bxinfoOCC1.reserve(3564);
	bxinfoOCC2.reserve(3564);
	for(size_t i=0;i<3564;++i){
	  bxinfoET.push_back(lumi::BunchCrossingInfo(lit->bxET.at(i).idx,lit->bxET.at(i).lumivalue,lit->bxET.at(i).lumierr,lit->bxET.at(i).lumiquality));
	  bxinfoOCC1.push_back(lumi::BunchCrossingInfo(lit->bxOCC1.at(i).idx,lit->bxOCC1.at(i).lumivalue,lit->bxOCC1.at(i).lumierr,lit->bxOCC1.at(i).lumiquality));
	  bxinfoOCC2.push_back(lumi::BunchCrossingInfo(lit->bxOCC2.at(i).idx,lit->bxOCC2.at(i).lumivalue,lit->bxOCC2.at(i).lumierr,lit->bxOCC2.at(i).lumiquality));
	}
	l->setBunchCrossingData(bxinfoET,lumi::ET);
	l->setBunchCrossingData(bxinfoOCC1,lumi::OCCD1);
	l->setBunchCrossingData(bxinfoOCC2,lumi::OCCD2);
	
	float deadfractionPerLS=-99.0;
	std::vector<lumi::TriggerInfo> triginfo;
	triginfo.reserve(192);
	lumi::TriggerInfo emt;
	std::fill_n(std::back_inserter(triginfo),192,emt);
	unsigned int deadcountPerLS=0;
	try{
	  deadcountPerLS=deadtime.at((lumisecid-1));
	  //std::cout<<"deadcountPer LS "<<deadcountPerLS<<std::endl;
	  deadfractionPerLS=deadcountPerLS*25.0*(1.0e-9)/93.244;
	  l->setDeadFraction(deadfractionPerLS);
	  std::vector<unsigned int> algobitcounts=algocount.at(lumisecid-1);
	  std::vector<unsigned int> techbitcounts=techcount.at(lumisecid-1);
	  for(size_t itrg=0; itrg<192; ++itrg){
	    triginfo.at(itrg).deadtimecount=deadcountPerLS;
	    if(itrg<128){
	      triginfo.at(itrg).prescale=algoprescale.at(itrg);
	      triginfo.at(itrg).name=algonames.at(itrg);
	      triginfo.at(itrg).triggercount=algobitcounts.at(itrg);
	      triginfo.at(itrg).deadtimecount=deadcountPerLS;
	    }else{
	      triginfo.at(itrg).prescale=techprescale.at(itrg-128);
	      triginfo.at(itrg).name=technames.at(itrg-128);
	      triginfo.at(itrg).triggercount=techbitcounts.at(itrg-128);
	      triginfo.at(itrg).deadtimecount=deadcountPerLS;
	    }
	  }
	}catch(const std::out_of_range& outOfRange){
	  std::cout<<"no trg found for LS "<<lumisecid<<std::endl;
	  l->setDeadFraction(deadfractionPerLS);
	}
	std::cout<<"\t deadfraction per LS "<<deadfractionPerLS<<std::endl;
	totaldeadtime+=deadcountPerLS;
	//std::cout<<"trigger size "<<triginfo.size()<<std::endl;
	l->setTriggerData(triginfo);
	l->print(std::cout);
	result.push_back(std::make_pair<lumi::LumiSectionData*,cond::Time_t>(l,current));
      }
      std::cout<<"total deadtime count "<<totaldeadtime<<std::endl;
      return std::string("mixedsource trurun ")+m_filename+";"+m_lumiversion;  
    }
  }
  return "";
}

DEFINE_EDM_PLUGIN(lumi::LumiRetrieverFactory,lumi::MixedSource,"mixedsource");
 

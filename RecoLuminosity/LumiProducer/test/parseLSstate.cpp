#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <stdexcept>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include "CoralBase/AttributeList.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/AttributeSpecification.h"
#include "CoralBase/Exception.h"
#include "CoralBase/TimeStamp.h"
#include "CoralBase/MessageStream.h"
#include "CoralKernel/Context.h"
#include "CoralKernel/IHandle.h"
#include "CoralKernel/IProperty.h"
#include "CoralKernel/IPropertyManager.h"
#include "RelationalAccess/AccessMode.h"
#include "RelationalAccess/ConnectionService.h"
#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/IConnectionServiceConfiguration.h"
#include "RelationalAccess/ITransaction.h"
#include "RelationalAccess/ITypeConverter.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ICursor.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ITable.h"
#include <boost/regex.hpp>
#include "RecoLuminosity/LumiProducer/interface/Utils.h"

namespace lumitest{
  void stringSplit(const std::string& instr, char delim, std::vector<std::string>&results){
    size_t cutAt=0;
    std::string str=instr;
    while( (cutAt=str.find_first_of(delim))!=std::string::npos){
      if(cutAt>0){
	results.push_back(str.substr(0,cutAt));
	str=str.substr(cutAt+1);
      }
    }
    if(str.length()>0){
      results.push_back(str);
    }
  }
  
  void fillMap(const std::vector<std::string>& inVector,
	       std::map<unsigned int,std::string>& result){
    std::vector<std::string>::const_iterator it;
    std::vector<std::string>::const_iterator itBeg=inVector.begin();
    std::vector<std::string>::const_iterator itEnd=inVector.end();
    for(it=itBeg;it!=itEnd;++it){
      //std::cout<<*it<<std::endl;
      std::string rundelimeterStr(it->begin(),it->end()-1);
      std::string stateStr(it->end()-1,it->end());
      //std::cout<<"rundelimeterStr "<<rundelimeterStr<<std::endl;
      float rundelimeter=0.0;
      if(!lumi::from_string(rundelimeter,rundelimeterStr,std::dec)){
	std::cout<<"failed to convert string to float"<<std::endl;
      }
      //std::cout<<"stateStr "<<stateStr<<std::endl;
      //
      //logic of rounding:
      //for states physics_declared T,F, use ceil function to round up then convert to unsigned int, because T,F will be set at the next LS boundary
      //for state paused P, use floor function to round down then convert to unsigned int, because we count the LS as paused as long as it contains pause (this logic could be changed)
      //
      if(stateStr=="P"){
	result.insert(std::make_pair((unsigned int)std::floor(rundelimeter),stateStr));}else if(stateStr=="T"||stateStr=="F"){
	result.insert(std::make_pair((unsigned int)std::ceil(rundelimeter),stateStr));
      }else{
	throw std::runtime_error("unknown LS state");
      }
    }
  }
}
int main(int argc,char** argv){
  unsigned int runnumber=0;
  if(!(runnumber=::atoi(argv[1]))){
    std::cout<<"must specify the run number"<<std::endl;
    return 0;
  }
  const boost::regex physicsE("%PHYSICS_DECLARED&(true|false|N/A)&(true|false|N/A)%");
  const boost::regex nameE("^CMS.LVL0:RUNSECTION_DELIMITER_DCSLHCFLAGS_([0-9]+)");
  boost::match_results<std::string::const_iterator> what;
  
  //
  //query runinfo db
  //
  //select name,string_value from runsession_parameter where  runnumber=:runnumber and (name like 'CMS.LVL0:RUNSECTION_DELIMITER_LS_%' or name like 'CMS.LVL0:RUNSECTION_DELIMITER_DCSLHCFLAGS_%') order by time; 
  //
  std::string serviceName("oracle://cms_omds_lb/CMS_RUNINFO");
  std::string authName("/afs/cern.ch/user/x/xiezhen/authentication.xml");
  std::string tabname("RUNSESSION_PARAMETER");
  try{
    coral::ConnectionService* conService = new coral::ConnectionService();
    coral::Context::instance().PropertyManager().property("AuthenticationFile")->set(authName);
    conService->configuration().setAuthenticationService("CORAL/Services/XMLAuthenticationService");
    conService->configuration().disablePoolAutomaticCleanUp();
    conService->configuration().setConnectionTimeOut(0);
    coral::MessageStream::setMsgVerbosity(coral::Error);
    coral::ISessionProxy* session = conService->connect( serviceName, coral::ReadOnly);
    coral::ITypeConverter& tpc=session->typeConverter();

    tpc.setCppTypeForSqlType("unsigned int","NUMBER(7)");
    tpc.setCppTypeForSqlType("unsigned int","NUMBER(10)");
    tpc.setCppTypeForSqlType("unsigned long long","NUMBER(20)");    
    
    coral::ITransaction& transaction=session->transaction();
    transaction.start(true); //true means readonly transaction
    coral::ISchema& schema=session->nominalSchema();
    if( !schema.existsTable(tabname) ){
      std::cout<<"table "<<tabname<<" doesn't exist"<<std::endl;
      return 0;
    }
    coral::IQuery* query=schema.tableHandle(tabname).newQuery();
    coral::AttributeList qoutput;
    qoutput.extend("NAME",typeid(std::string));
    qoutput.extend("STRING_VALUE",typeid(std::string));    
    coral::AttributeList qcondition;
    qcondition.extend("runnumber",typeid(unsigned int));
    qcondition.extend("delimiterls",typeid(std::string));
    qcondition.extend("dcslhcflag",typeid(std::string));
    qcondition["runnumber"].data<unsigned int>()=runnumber;
    qcondition["delimiterls"].data<std::string>()="CMS.LVL0:RUNSECTION_DELIMITER_LS_%";
    qcondition["dcslhcflag"].data<std::string>()="CMS.LVL0:RUNSECTION_DELIMITER_DCSLHCFLAGS_%";
    query->addToOutputList("NAME");
    query->addToOutputList("STRING_VALUE");
    query->setCondition("RUNNUMBER =:runnumber AND (NAME like :delimiterls OR NAME like :dcslhcflag)",qcondition);
    query->addToOrderList("TIME");
    query->defineOutput(qoutput);
    coral::ICursor& cursor=query->execute();
    while( cursor.next() ){
      const coral::AttributeList& row=cursor.currentRow();
      //row.toOutputStream(std::cout)<<std::endl;
      std::string name=row["NAME"].data<std::string>();
      std::string value=row["STRING_VALUE"].data<std::string>();
      std::cout<<"name: "<<name<<", value: "<<value<<std::endl;
      boost::regex_match(name,what,nameE,boost::match_default);
      std::string statChar;
      if(what[0].matched){
	if(value=="null"){
	  statChar="P";//is a pause
	}else{
	  boost::regex_search(value,what,physicsE,boost::match_default);
	  if(what[0].matched){
	    std::string operatorBitValue=std::string(what[2].first,what[2].second);
	    if(operatorBitValue=="true"){
	      statChar="T";
	    }else{
	      statChar="F";
	    }
	  }
	}
      }
    }
    delete query;
    transaction.commit();
    delete session;
    delete conService;
  }catch(const std::exception& er){
    std::cout<<"caught exception "<<er.what()<<std::endl;
    throw er;
  }
  const boost::regex e("%*PHYSICS_DECLARED&(true)|(false)|(N/A)&(true)|(false)|(N/A)%");
  
  // float number as string for runsection delimiter,
  // "T" for true,"F" for false, "P" for pause
  //
  // the output of the parsing are 2 booleans per lumisection
  //
  // physicsDeclared
  // isPaused
  //
  // a lumisection can be considered for recorded luminosity only if
  // PhysicsDeclared && !isPaused
  // 
  // source of this decision is documented here:
  // https://savannah.cern.ch/support/?112921
  //
  std::string LSstateInputstr("1.0T,19.9P,21.6F,23.54P");//output of database query in one string format
  unsigned int totalLS=40; //suppose there are 40 cms ls, I want to assign T,F,P state to each of them
  std::vector<std::string> parseresult;
  lumitest::stringSplit(LSstateInputstr,',',parseresult);//split the input string into a vector of string pairs by ','
  std::map<unsigned int,std::string> delimiterMap;
  lumitest::fillMap(parseresult,delimiterMap);//parse the vector into state boundary LS(key) to LS state(value) map
  //keys [1,19,22,23]
  for(unsigned int ls=1;ls<=totalLS;++ls){ //loop over my LS comparing it to the state boundaries
    std::map<unsigned int,std::string>::const_iterator lsItUp;
    lsItUp=delimiterMap.upper_bound(ls);
    std::string r;
    if(lsItUp!=delimiterMap.end()){
      lsItUp=delimiterMap.upper_bound(ls);
      --lsItUp;
      r=(*lsItUp).second;
      //std::cout<<"LS "<<ls<<std::endl;
      //std::cout<<"boundary "<<(*lsItUp).first<<std::endl;
      //std::cout<<"state "<<r<<std::endl;
    }else{
      std::map<unsigned int,std::string>::reverse_iterator lsItLast=delimiterMap.rbegin();    
      r=(*lsItLast).second;
      //std::cout<<"LS "<<ls<<std::endl;
      //std::cout<<"boundary "<<(*lsItLast).first<<std::endl;
      //std::cout<<"state "<<r<<std::endl;
    }
    std::cout<<"LS : "<<ls<<" , state : "<<r<<std::endl; 
  }
}

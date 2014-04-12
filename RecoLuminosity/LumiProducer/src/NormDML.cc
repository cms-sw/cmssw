#include "RecoLuminosity/LumiProducer/interface/NormDML.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ICursor.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/AttributeSpecification.h"
#include "RecoLuminosity/LumiProducer/interface/LumiNames.h"
#include <algorithm>
#include <map>
#include <iostream>
#include <sstream>
#include <boost/algorithm/string.hpp>
#include <boost/tokenizer.hpp>
lumi::NormDML::NormDML(){
}
unsigned long long 
lumi::NormDML::normIdByName(const coral::ISchema& schema,const std::string& normtagname){
  ///select max(DATA_ID) FROM LUMINORMSV2 WHERE ENTRY_NAME=:normname
  unsigned long long result=0;
  std::vector<unsigned long long> luminormids;
  coral::IQuery* qHandle=schema.newQuery();
  qHandle->addToTableList( lumi::LumiNames::luminormv2TableName() );
  qHandle->addToOutputList("DATA_ID");
  if(!normtagname.empty()){
    std::string qConditionStr("ENTRY_NAME=:normtagname");
    coral::AttributeList qCondition;
    qCondition.extend("normtagname",typeid(std::string));
    qCondition["normtagname"].data<std::string>()=normtagname;
    qHandle->setCondition(qConditionStr,qCondition);
  }
  coral::AttributeList qResult;
  qResult.extend("DATA_ID",typeid(unsigned long long));
  qHandle->defineOutput(qResult);
  coral::ICursor& cursor=qHandle->execute();
  while( cursor.next() ){
    const coral::AttributeList& row=cursor.currentRow();
    luminormids.push_back(row["DATA_ID"].data<unsigned long long>());
  }
  delete qHandle;
  std::vector<unsigned long long>::iterator resultIt;
  for(resultIt=luminormids.begin();resultIt!=luminormids.end();++resultIt){
    if( (*resultIt)>result){
      result=*resultIt;
    }
  }
  return result;
}
void
lumi::NormDML::normIdByType(const coral::ISchema& schema,std::map<std::string,unsigned long long>& resultMap,lumi::NormDML::LumiType lumitype,bool defaultonly){
  ///select max(DATA_ID) FROM LUMINORMSV2 WHERE LUMITYPE=:lumitype and ISTYPEDEFAULT=1
  coral::IQuery* qHandle=schema.newQuery();
  qHandle->addToTableList( lumi::LumiNames::luminormv2TableName() );
  qHandle->addToOutputList("DATA_ID");
  qHandle->addToOutputList("ENTRY_NAME");
  coral::AttributeList qCondition;
  std::string qConditionStr("LUMITYPE=:lumitype");
  qCondition.extend("lumitype",typeid(std::string));
  std::string lumitypeStr("HF");
  if(lumitype!=lumi::NormDML::HF){
    lumitypeStr="PIXEL";
  }
  qCondition["lumitype"].data<std::string>()=lumitypeStr;
  if(defaultonly){
    qConditionStr+=" AND ISTYPEDEFAULT=:istypedefault";
    qCondition.extend("istypedefault",typeid(unsigned int));
    qCondition["istypedefault"].data<unsigned int>()=1;
  }
  qHandle->setCondition(qConditionStr,qCondition);
  coral::AttributeList qResult;
  qResult.extend("DATA_ID",typeid(unsigned long long));
  qResult.extend("ENTRY_NAME",typeid(std::string));
  qHandle->defineOutput(qResult);
  try{
    coral::ICursor& cursor=qHandle->execute();
    while( cursor.next() ){
      const coral::AttributeList& row=cursor.currentRow();
      const std::string normname=row["ENTRY_NAME"].data<std::string>();
      unsigned long long normid=row["DATA_ID"].data<unsigned long long>();
      if(resultMap.find(normname)==resultMap.end()){
	resultMap.insert(std::make_pair(normname,normid));
      }else{
	if(resultMap[normname]<normid){
	  resultMap.insert(std::make_pair(normname,normid));
	}
      }
    }
  }catch(const coral::Exception& er){
    std::cout<<"database error in NormDML::normIdByType "<<er.what()<<std::endl;
    delete qHandle;
    throw er;
  }catch(...){
    throw;
  }
  delete qHandle;
}
void
lumi::NormDML::normById(const coral::ISchema&schema, 
			unsigned long long normid, 
			std::map< unsigned int,lumi::NormDML::normData >& result){
  ///select * from luminormsv2data where data_id=normid
  coral::IQuery* qHandle=schema.newQuery();
  qHandle->addToTableList( lumi::LumiNames::luminormv2dataTableName() );  
  std::string qConditionStr("DATA_ID=:normid ");
  coral::AttributeList qCondition;
  qCondition.extend("normid",typeid(unsigned long long));
  qCondition["normid"].data<unsigned long long>()=normid;
  qHandle->setCondition(qConditionStr,qCondition);
  coral::AttributeList qResult;
  coral::ICursor& cursor=qHandle->execute();
  while( cursor.next() ){
    const coral::AttributeList& row=cursor.currentRow();
    unsigned int since=row["SINCE"].data<unsigned int>();
    if(result.find(since)==result.end()){
      lumi::NormDML::normData thisnorm;
      result.insert(std::make_pair(since,thisnorm));
    }
    const std::string correctorStr=row["CORRECTOR"].data<std::string>();
    if(!row["AMODETAG"].isNull()){
      result[since].amodetag=row["AMODETAG"].data<std::string>();
    }
    if(!row["NOMINALEGEV"].isNull()){
      result[since].beamegev=row["NOMINALEGEV"].data<unsigned int>();
    }
   
    std::vector<std::string> correctorParams;
    parseLumiCorrector(correctorStr,correctorParams);
    result[since].corrfunc=*(correctorParams.begin());
    for(std::vector<std::string>::iterator corrIt=correctorParams.begin()+1;
	corrIt!=correctorParams.end();corrIt++){
      std::string paramName=boost::to_upper_copy(*corrIt);
      if(paramName==std::string("AFTERGLOW")){
	const std::string afterglowStr=row["AFTERGLOW"].data<std::string>();
	parseAfterglows(afterglowStr,result[since].afterglows);
      }else{
	float param=row[paramName].data<float>();
	result[since].coefficientmap.insert(std::make_pair(paramName,param));
      }
    }
  }
  delete qHandle;
}
void
lumi::NormDML::parseLumiCorrector(const std::string& correctorStr,
		     std::vector<std::string>& correctorParams){
  std::string cleancorrectorStr(correctorStr);
  boost::trim(cleancorrectorStr);
  boost::split(correctorParams,cleancorrectorStr,boost::is_any_of(":,")); 
}
void
lumi::NormDML::parseAfterglows(const std::string& afterglowStr,std::map<unsigned int,float>& afterglowmap){
  typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
  boost::char_separator<char> sep("[(,)] ");
  tokenizer tokens(afterglowStr,sep);
  unsigned int counter=1;
  std::string thresholdStr;
  unsigned int threshold;
  for(tokenizer::iterator tok_iter=tokens.begin();tok_iter != tokens.end(); ++tok_iter){
    if(counter%2==0){
      std::string valStr=*(tok_iter);
      float val=0.;
      std::stringstream strStream(valStr);
      strStream>>val;
      afterglowmap.insert(std::make_pair(threshold,val));
    }else{
      thresholdStr=*(tok_iter);
      std::stringstream strStream(thresholdStr);
      strStream>>threshold;
    }    
    ++counter;
  }
}

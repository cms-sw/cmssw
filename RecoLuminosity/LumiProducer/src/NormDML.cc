#include "RecoLuminosity/LumiProducer/interface/NormDML.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ICursor.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/AttributeSpecification.h"
#include "RecoLuminosity/LumiProducer/interface/LumiNames.h"
#include <algorithm>
#include <utility>
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
    std::string qConditionStr("ENTRY_NAME=:normtagname ");
    coral::AttributeList qCondition;
    qCondition.extend("normtagname",typeid(std::string));
    qCondition["normtagname"].data<std::string>()=normtagname;
    qHandle->setCondition(qConditionStr,qCondition);
  }
  coral::AttributeList qResult;
  qResult.extend("DATA_ID",typeid("unsigned long long"));
  qHandle->defineOutput(qResult);
  
  coral::ICursor& cursor=qHandle->execute();
  while( cursor.next() ){
    const coral::AttributeList& row=cursor.currentRow();
    luminormids.push_back(row["DATA_ID"].data<unsigned long long>());
  }
  delete qHandle;
  if(luminormids.size() !=0){
    std::vector<unsigned long long>::iterator resultIt=std::max_element( luminormids.begin(), luminormids.end());
    return *resultIt;
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
  if(lumitype){
    qConditionStr+=" AND ISTYPEDEFAULT=:istypedefault";
    qCondition.extend("istypedefault",typeid(unsigned int));
    qCondition["istypedefault"].data<unsigned int>()=1;
  }
  qHandle->setCondition(qConditionStr,qCondition);
  coral::AttributeList qResult;
  qResult.extend("DATA_ID",typeid("unsigned long long"));
  qResult.extend("ENTRY_NAME",typeid("std::string"));
  qHandle->defineOutput(qResult);
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
  delete qHandle;
}
void
lumi::NormDML::normById(const coral::ISchema&schema, unsigned long long normid, std::map< unsigned int,lumi::NormDML::normData >& result)const{
  ///select * from luminormsv2data where data_id=normid

}

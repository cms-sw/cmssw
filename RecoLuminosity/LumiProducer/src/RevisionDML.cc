#include "RecoLuminosity/LumiProducer/interface/RevisionDML.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/ITableDataEditor.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ICursor.h"
#include "RelationalAccess/SchemaException.h" 
#include "CoralBase/AttributeList.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/AttributeSpecification.h"
#include "CoralBase/TimeStamp.h"
#include "RecoLuminosity/LumiProducer/interface/LumiNames.h"
#include "RecoLuminosity/LumiProducer/interface/idDealer.h"
#include "RecoLuminosity/LumiProducer/interface/Exception.h"
#include <algorithm>
unsigned long long 
lumi::RevisionDML::getEntryInBranchByName(coral::ISchema& schema,
					  const std::string& datatableName,
					  const std::string& entryname,
					  const std::string& branchname){
  unsigned long long entry_id=0;
  coral::IQuery* qHandle=schema.newQuery();
  qHandle->addToTableList( lumi::LumiNames::entryTableName(datatableName),"e");
  qHandle->addToTableList( lumi::LumiNames::revisionTableName(),"r");
  qHandle->addToOutputList("e.ENTRY_ID","entry_id");

  coral::AttributeList qCondition;
  qCondition.extend("entryname",typeid(std::string));
  qCondition.extend("branchname",typeid(std::string));
  qCondition["entryname"].data<std::string>()=entryname;
  qCondition["branchname"].data<std::string>()=branchname;
  std::string qConditionStr("r.REVISION_ID=e.REVISION_ID and e.NAME=:entryname AND r.BRANCH_NAME=:branchname");

  coral::AttributeList qResult;
  qResult.extend("entry_id",typeid(unsigned long long));
  qHandle->defineOutput(qResult);
  qHandle->setCondition(qConditionStr,qCondition);
  coral::ICursor& cursor=qHandle->execute();
  while(cursor.next()){
    entry_id=cursor.currentRow()["entry_id"].data<unsigned long long>();
  }
  delete qHandle;
  return entry_id;
}
void 
lumi::RevisionDML::bookNewEntry(coral::ISchema& schema,
				const std::string& datatableName,
				lumi::RevisionDML::Entry& entry){
  lumi::idDealer idg(schema);
  const std::string entrytableName=lumi::LumiNames::entryTableName(datatableName);
  entry.revision_id=idg.generateNextIDForTable(lumi::LumiNames::revisionTableName());
  entry.data_id=idg.generateNextIDForTable(datatableName);
  entry.entry_id=idg.generateNextIDForTable(entrytableName);
}
void 
lumi::RevisionDML::bookNewRevision(coral::ISchema& schema,
				   const std::string& datatableName,
				   lumi::RevisionDML::Entry& revision){
  lumi::idDealer idg(schema);
  revision.revision_id=idg.generateNextIDForTable(lumi::LumiNames::revisionTableName());
  revision.data_id=idg.generateNextIDForTable(datatableName);
}
void 
lumi::RevisionDML::addEntry(coral::ISchema& schema,
			    const std::string& datatableName,
			    const lumi::RevisionDML::Entry& entry,
			    unsigned long long branchid,
			    const std::string& branchname ){
  coral::AttributeList revdata;
  revdata.extend("REVISION_ID",typeid(unsigned long long));
  revdata.extend("BRANCH_ID",typeid(unsigned long long));
  revdata.extend("BRANCH_NAME",typeid(std::string));
  revdata.extend("CTIME",typeid(coral::TimeStamp));
  revdata["REVISION_ID"].data<unsigned long long>()=entry.revision_id;
  revdata["BRANCH_ID"].data<unsigned long long>()=branchid;
  revdata["BRANCH_NAME"].data<std::string>()=branchname;
  revdata["CTIME"].data<coral::TimeStamp>()=coral::TimeStamp::now();
  const std::string revTableName=lumi::LumiNames::revisionTableName();
  schema.tableHandle(revTableName).dataEditor().insertRow(revdata);

  coral::AttributeList entrydata;
  entrydata.extend("REVISION_ID",typeid(unsigned long long));
  entrydata.extend("ENTRY_ID",typeid(unsigned long long));
  entrydata.extend("NAME",typeid(std::string));
  entrydata["REVISION_ID"].data<unsigned long long>()=entry.revision_id;
  entrydata["ENTRY_ID"].data<unsigned long long>()=entry.entry_id;
  entrydata["NAME"].data<std::string>()=entry.entry_name;
  const std::string entryTableName=lumi::LumiNames::entryTableName(datatableName);
  schema.tableHandle(entryTableName).dataEditor().insertRow(entrydata);
  
  coral::AttributeList revmapdata;
  revmapdata.extend("REVISION_ID",typeid(unsigned long long));
  revmapdata.extend("DATA_ID",typeid(unsigned long long));

  revmapdata["REVISION_ID"].data<unsigned long long>()=entry.revision_id;
  revmapdata["DATA_ID"].data<unsigned long long>()=entry.data_id;

  const std::string revmapTableName=lumi::LumiNames::revmapTableName(datatableName);
  schema.tableHandle(revmapTableName).dataEditor().insertRow(revmapdata);
}
void 
lumi::RevisionDML::addRevision(coral::ISchema& schema,
			       const std::string& datatableName,
			       const lumi::RevisionDML::Entry& revision,
			       unsigned long long branchid,
			       std::string& branchname ){
  coral::AttributeList revdata;
  revdata.extend("REVISION_ID",typeid(unsigned long long));
  revdata.extend("BRANCH_ID",typeid(unsigned long long));
  revdata.extend("BRANCH_NAME",typeid(std::string));
  revdata.extend("CTIME",typeid(coral::TimeStamp));
  revdata["REVISION_ID"].data<unsigned long long>()=revision.revision_id;
  revdata["BRANCH_ID"].data<unsigned long long>()=branchid;
  revdata["BRANCH_NAME"].data<std::string>()=branchname;
  revdata["CTIME"].data<coral::TimeStamp>()=coral::TimeStamp::now();
  schema.tableHandle(lumi::LumiNames::revisionTableName()).dataEditor().insertRow(revdata);
  coral::AttributeList revmapdata;
  revmapdata.extend("REVISION_ID",typeid(unsigned long long));
  revmapdata.extend("DATA_ID",typeid(unsigned long long));
  revmapdata["REVISION_ID"].data<unsigned long long>()=revision.revision_id;
  revmapdata["DATA_ID"].data<unsigned long long>()=revision.data_id;
  const std::string revmapTableName=lumi::LumiNames::revmapTableName(datatableName);
  schema.tableHandle(revmapTableName).dataEditor().insertRow(revmapdata);
}
void 
lumi::RevisionDML::insertLumiRunData(coral::ISchema& schema,
				     const lumi::RevisionDML::LumiEntry& lumientry){
  coral::AttributeList lumirundata;
  lumirundata.extend("DATA_ID",typeid(unsigned long long));
  lumirundata.extend("ENTRY_ID",typeid(unsigned long long));
  lumirundata.extend("ENTRY_NAME",typeid(std::string));
  lumirundata.extend("RUNNUM",typeid(unsigned int));
  lumirundata.extend("SOURCE",typeid(std::string));
  lumirundata.extend("NOMINALEGEV",typeid(float));
  lumirundata.extend("NCOLLIDINGBUNCHES",typeid(unsigned int));
  lumirundata["DATA_ID"].data<unsigned long long>()=lumientry.data_id;
  lumirundata["ENTRY_ID"].data<unsigned long long>()=lumientry.entry_id;
  lumirundata["ENTRY_NAME"].data<std::string>()=lumientry.entry_name;
  lumirundata["RUNNUM"].data<unsigned int>()=lumientry.runnumber;
  lumirundata["SOURCE"].data<std::string>()=lumientry.source;
  lumirundata["NOMINALEGEV"].data<float>()=lumientry.bgev;
  lumirundata["NCOLLIDINGBUNCHES"].data<unsigned int>()=lumientry.ncollidingbunches;
  const std::string lumidataTableName=lumi::LumiNames::lumidataTableName();
  schema.tableHandle(lumidataTableName).dataEditor().insertRow(lumirundata);
}
void 
lumi::RevisionDML::insertTrgRunData(coral::ISchema& schema,
				    const lumi::RevisionDML::TrgEntry& trgentry){
  coral::AttributeList trgrundata;
  trgrundata.extend("DATA_ID",typeid(unsigned long long));
  trgrundata.extend("ENTRY_ID",typeid(unsigned long long));
  trgrundata.extend("ENTRY_NAME",typeid(std::string));
  trgrundata.extend("RUNNUM",typeid(unsigned int));
  trgrundata.extend("SOURCE",typeid(std::string));
  trgrundata.extend("BITZERONAME",typeid(std::string));
  trgrundata.extend("BITNAMECLOB",typeid(std::string));
  trgrundata["DATA_ID"].data<unsigned long long>()=trgentry.data_id;
  trgrundata["ENTRY_ID"].data<unsigned long long>()=trgentry.entry_id;
  trgrundata["ENTRY_NAME"].data<std::string>()=trgentry.entry_name;
  trgrundata["RUNNUM"].data<unsigned int>()=trgentry.runnumber;
  trgrundata["SOURCE"].data<std::string>()=trgentry.source;
  trgrundata["BITZERONAME"].data<std::string>()=trgentry.bitzeroname;
  trgrundata["BITNAMECLOB"].data<std::string>()=trgentry.bitnames;
  const std::string trgdataTableName=lumi::LumiNames::trgdataTableName();
  schema.tableHandle(trgdataTableName).dataEditor().insertRow(trgrundata);
}
void 
lumi::RevisionDML::insertHltRunData(coral::ISchema& schema,
				    const lumi::RevisionDML::HltEntry& hltentry){
  coral::AttributeList hltrundata;
  hltrundata.extend("DATA_ID",typeid(unsigned long long));
  hltrundata.extend("ENTRY_ID",typeid(unsigned long long));
  hltrundata.extend("ENTRY_NAME",typeid(std::string));
  hltrundata.extend("RUNNUM",typeid(unsigned int));
  hltrundata.extend("SOURCE",typeid(std::string));
  hltrundata.extend("NPATH",typeid(unsigned int));
  hltrundata.extend("PATHNAMECLOB",typeid(std::string));
  hltrundata["DATA_ID"].data<unsigned long long>()=hltentry.data_id;
  hltrundata["ENTRY_ID"].data<unsigned long long>()=hltentry.entry_id;
  hltrundata["ENTRY_NAME"].data<std::string>()=hltentry.entry_name;
  hltrundata["RUNNUM"].data<unsigned int>()=hltentry.runnumber;
  hltrundata["SOURCE"].data<std::string>()=hltentry.source;
  hltrundata["NPATH"].data<unsigned int>()=hltentry.npath;
  hltrundata["PATHNAMECLOB"].data<std::string>()=hltentry.pathnames;
  const std::string hltdataTableName=lumi::LumiNames::hltdataTableName();
  schema.tableHandle(hltdataTableName).dataEditor().insertRow(hltrundata);
}

unsigned long long
lumi::RevisionDML::currentHFDataTagId(coral::ISchema& schema){
  unsigned long long currentdatatagid=0;
  std::vector<unsigned long long> alltagids;
  coral::IQuery* qHandle=schema.newQuery();
  qHandle->addToTableList( lumi::LumiNames::tagsTableName());
  qHandle->addToOutputList("TAGID");
  coral::AttributeList qResult;
  qResult.extend("TAGID",typeid(unsigned long long));
  qHandle->defineOutput(qResult);
  coral::ICursor& cursor=qHandle->execute();
  while(cursor.next()){
    if(!cursor.currentRow()["TAGID"].isNull()){
      alltagids.push_back(cursor.currentRow()["TAGID"].data<unsigned long long>());
    }
  }
  delete qHandle;
  if(alltagids.size()>0){
    std::vector<unsigned long long>::iterator currentdatatagidIt=std::max_element(alltagids.begin(),alltagids.end());
    currentdatatagid=*currentdatatagidIt;
  }
  return currentdatatagid;
}

unsigned long long
lumi::RevisionDML::HFDataTagIdByName(coral::ISchema& schema,
				     const std::string& datatagname){
  unsigned long long datatagid=0;
  coral::IQuery* qHandle=schema.newQuery();
  qHandle->addToTableList( lumi::LumiNames::tagsTableName());
  const std::string conditionStr("TAGNAME=:tagname");
  coral::AttributeList condition;
  condition.extend("tagname",typeid(std::string));
  condition["tagname"].data<std::string>()=datatagname;
  qHandle->addToOutputList("TAGID");
  coral::AttributeList qResult;
  qResult.extend("TAGID",typeid(unsigned long long));
  qHandle->setCondition(conditionStr,condition);
  qHandle->defineOutput(qResult);
  coral::ICursor& cursor=qHandle->execute();
  while(cursor.next()){
    if(!cursor.currentRow()["TAGID"].isNull()){
      datatagid=cursor.currentRow()["TAGID"].data<unsigned long long>();
    }
  }
  delete qHandle;
  return datatagid;
}

unsigned long long
lumi::RevisionDML::addRunToCurrentHFDataTag(coral::ISchema& schema,
					    unsigned int runnum,
					    unsigned long long lumiid,
					    unsigned long long trgid,
					    unsigned long long hltid,
					    const std::string& patchcomment)
{
  unsigned long long currenttagid=currentHFDataTagId(schema);  
  coral::AttributeList tagrundata;
  tagrundata.extend("TAGID",typeid(unsigned long long));
  tagrundata.extend("RUNNUM",typeid(unsigned int));
  tagrundata.extend("LUMIDATAID",typeid(unsigned long long));
  tagrundata.extend("TRGDATAID",typeid(unsigned long long));
  tagrundata.extend("HLTDATAID",typeid(unsigned long long));
  tagrundata.extend("CREATIONTIME",typeid(coral::TimeStamp));
  tagrundata.extend("COMMENT",typeid(std::string));
  tagrundata["TAGID"].data<unsigned long long>()=currenttagid;
  tagrundata["RUNNUM"].data<unsigned int>()=runnum;
  tagrundata["LUMIDATAID"].data<unsigned long long>()=lumiid;
  tagrundata["TRGDATAID"].data<unsigned long long>()=trgid;
  tagrundata["HLTDATAID"].data<unsigned long long>()=hltid;
  tagrundata["CREATIONTIME"].data<coral::TimeStamp>()=coral::TimeStamp::now();
  tagrundata["COMMENT"].data<std::string>()=patchcomment;
  const std::string tagrunTableName=lumi::LumiNames::tagRunsTableName();
  try{
    schema.tableHandle(lumi::LumiNames::tagRunsTableName()).dataEditor().insertRow(tagrundata);
  }catch(const coral::DuplicateEntryInUniqueKeyException& er){
    throw lumi::duplicateRunInDataTagException("","addRunToCurrentHFDataTag","RevisionDML");
  }
  return currenttagid;
}

lumi::RevisionDML::DataID
lumi::RevisionDML::dataIDForRun(coral::ISchema& schema,
			  unsigned int runnum,
			  unsigned long long tagid){
  lumi::RevisionDML::DataID result;
  coral::IQuery* qHandle=schema.newQuery();
  qHandle->addToTableList( lumi::LumiNames::tagRunsTableName());
  qHandle->addToOutputList("LUMIDATAID");
  qHandle->addToOutputList("TRGDATAID");
  qHandle->addToOutputList("HLTDATAID");
  coral::AttributeList qResult;
  qResult.extend("LUMIDATAID",typeid(unsigned long long));
  qResult.extend("TRGDATAID",typeid(unsigned long long));
  qResult.extend("HLTDATAID",typeid(unsigned long long));
  qHandle->defineOutput(qResult);
  coral::AttributeList qCondition;
  qCondition.extend("tagid",typeid(unsigned long long));
  qCondition.extend("runnum",typeid(unsigned int));
  qCondition["tagid"].data<unsigned long long>()=tagid;
  qCondition["runnum"].data<unsigned int>()=runnum;
  std::string qConditionStr("TAGID<=:tagid AND RUNNUM=:runnum");
  qHandle->setCondition(qConditionStr,qCondition);
  coral::ICursor& cursor=qHandle->execute();
  unsigned long long minlumid=0;
  unsigned long long mintrgid=0;
  unsigned long long minhltid=0;
  while(cursor.next()){
    if(!cursor.currentRow()["LUMIDATAID"].isNull()){
      unsigned long long lumiid=cursor.currentRow()["LUMIDATAID"].data<unsigned long long>();      
      if(lumiid>minlumid){
	result.lumi_id=lumiid;
      }
      
    }
    if(!cursor.currentRow()["TRGDATAID"].isNull()){
      unsigned long long trgid=cursor.currentRow()["TRGDATAID"].data<unsigned long long>();      
      if(trgid>mintrgid){
	result.trg_id=trgid;
      }
    }
    if(!cursor.currentRow()["HLTDATAID"].isNull()){
      unsigned long long hltid=cursor.currentRow()["HLTDATAID"].data<unsigned long long>();  
      if(hltid>minhltid){
	result.hlt_id=hltid;
      }
    }
  }
  delete qHandle;
  return result;
}

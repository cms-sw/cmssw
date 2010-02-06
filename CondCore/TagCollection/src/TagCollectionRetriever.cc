//
// Package:    CondCore/TagCollection
// Class:      TagCollectionRetriever
//
// Author:      Zhen Xie
//
#include "CondCore/TagCollection/interface/TagCollectionRetriever.h"
#include "CondCore/TagCollection/interface/TagDBNames.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ICursor.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/Attribute.h"
#include "RelationalAccess/SchemaException.h"
//#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/TagCollection/interface/Exception.h"

//#include <iostream>
cond::TagCollectionRetriever::TagCollectionRetriever( cond::DbSession& coraldb ):m_coraldb(coraldb){
}
cond::TagCollectionRetriever::~TagCollectionRetriever(){}

void 
cond::TagCollectionRetriever::getTagCollection( const std::string& globaltag,
                                                std::set<cond::TagMetadata >& result){
  if(!m_coraldb.nominalSchema().existsTable(cond::tagInventoryTable)){
    throw cond::nonExistentGlobalTagInventoryException("TagCollectionRetriever::getTagCollection");
  }
  std::pair<std::string,std::string> treenodepair=parseglobaltag(globaltag);
  std::string treename=treenodepair.first;
  std::string nodename=treenodepair.second;
  //std::cout<<"treename "<<treename<<std::endl;
  //std::cout<<"nodename "<<nodename<<std::endl;
  std::string treetablename(cond::tagTreeTablePrefix);
  if( !treename.empty() ){
    for(unsigned int i=0; i<treename.size(); ++i){
      treename[i]=std::toupper(treename[i]);	
    }
    treetablename+="_";
    treetablename+=treename;
  }
  if( !m_coraldb.nominalSchema().existsTable(treetablename) ){
    throw cond::nonExistentGlobalTagException("TagCollectionRetriever::getTagCollection",globaltag);
  }
  coral::IQuery* query=m_coraldb.nominalSchema().newQuery();
  //std::cout<<"treetablename "<<treetablename<<std::endl;
  query->addToTableList( treetablename, "p1" );
  query->addToTableList( treetablename, "p2" );
  query->addToOutputList( "p1.tagid" );
  query->setRowCacheSize( 100 );
  coral::AttributeList bindData;
  bindData.extend( "nodelabel",typeid(std::string) );
  bindData.extend( "tagid",typeid(unsigned int) );
  bindData["tagid"].data<unsigned int>()=0;
  bindData["nodelabel"].data<std::string>()=nodename;
  query->setCondition( "p1.lft BETWEEN p2.lft AND p2.rgt AND p2.nodelabel = :nodelabel AND p1.tagid <> :tagid", bindData );
  coral::AttributeList qresult;
  qresult.extend("tagid", typeid(unsigned int));
  query->defineOutput(qresult);
  std::vector<unsigned int> leaftagids;
  leaftagids.reserve(100);
  coral::ICursor& cursor = query->execute();
  while( cursor.next() ) {
    const coral::AttributeList& row = cursor.currentRow();
    leaftagids.push_back(row["tagid"].data<unsigned int>());
  }
  cursor.close();
  delete query;
  std::vector<unsigned int>::iterator it;
  std::vector<unsigned int>::iterator itBeg=leaftagids.begin();
  std::vector<unsigned int>::iterator itEnd=leaftagids.end();
  coral::ITable& tagInventorytable=m_coraldb.nominalSchema().tableHandle(cond::tagInventoryTable);
  for( it=itBeg; it!=itEnd; ++it ){
    coral::IQuery* leaftagquery=tagInventorytable.newQuery();
    leaftagquery->addToOutputList( "tagname" );
    leaftagquery->addToOutputList( "pfn" );
    leaftagquery->addToOutputList( "recordname" );
    leaftagquery->addToOutputList( "objectname" );
    leaftagquery->addToOutputList( "labelname" );
    coral::AttributeList myresult;
    myresult.extend("tagname",typeid(std::string));
    myresult.extend("pfn",typeid(std::string));
    myresult.extend("recordname",typeid(std::string));
    myresult.extend("objectname",typeid(std::string));
    myresult.extend("labelname",typeid(std::string));
    leaftagquery->defineOutput( myresult );
    coral::AttributeList bindVariableList;
    bindVariableList.extend("tagid",typeid(unsigned int));
    leaftagquery->setCondition( "tagid = :tagid",bindVariableList );
    leaftagquery->limitReturnedRows(1,0);
    bindVariableList["tagid"].data<unsigned int>()=*it;
    coral::ICursor& cursor2 =leaftagquery->execute();
    if( cursor2.next() ){
      const coral::AttributeList& row = cursor2.currentRow();
      cond::TagMetadata tagmetadata;
      std::string tagname=row["tagname"].data<std::string>();
      tagmetadata.tag=tagname;
      tagmetadata.pfn=row["pfn"].data<std::string>();
      tagmetadata.recordname=row["recordname"].data<std::string>();
      tagmetadata.objectname=row["objectname"].data<std::string>();
      tagmetadata.labelname=row["labelname"].data<std::string>();
      if(! result.insert(tagmetadata).second ){
	throw cond::Exception("cond::TagCollectionRetriever::getTagCollection tag "+tagname+" from "+tagmetadata.pfn+" already exist, cannot insert in the tag collection ");
      }
    }
    cursor2.close();
    delete leaftagquery;
  }
}

std::pair<std::string,std::string>
cond::TagCollectionRetriever::parseglobaltag(const std::string& globaltag){
  std::pair<std::string,std::string> result;
  std::size_t pos=globaltag.find("::");
  if(pos==std::string::npos){
    result.first="";
    result.second=globaltag;
  }else{
    result.first=globaltag.substr(0,pos);
    result.second=globaltag.substr(pos+2);
  }
  return result;
}


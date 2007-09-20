#include "TagCollectionRetriever.h"
#include "TagDBNames.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ICursor.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/Exception.h"
#include "CoralBase/AttributeSpecification.h"
#include "CondCore/DBCommon/interface/CoralTransaction.h"

cond::TagCollectionRetriever::TagCollectionRetriever( cond::CoralTransaction& coraldb ):m_coraldb(&coraldb){}
cond::TagCollectionRetriever::~TagCollectionRetriever(){}
void 
cond::TagCollectionRetriever::getTagCollection( const std::string& roottag,
		       std::map< std::string, cond::TagMetadata >& result){
  coral::ITable& tagInventorytable=m_coraldb->nominalSchema().tableHandle(cond::TagDBNames::tagInventoryTable());
  coral::ITable& tagTreetable=m_coraldb->nominalSchema().tableHandle(cond::TagDBNames::tagTreeTable());
  //SELECT p1.tagid FROM treetable AS p1, treetable AS p2 WHERE p1.lft BETWEEN p2.lft AND p2.rgt AND p2.nodelabel=%s AND p1.tagid <> 0 ORDER BY p1.lft ASC
  coral::IQuery* query=tagTreetable.newQuery();
  query->addToTableList( cond::TagDBNames::tagTreeTable(), "p1" );
  query->addToTableList( cond::TagDBNames::tagTreeTable(), "p2" );
  query->addToOutputList( "p1.tagid" );
  query->setRowCacheSize( 100 );
  coral::AttributeList bindData;
  bindData.extend( "tagid",typeid(unsigned int) );
  bindData["tagid"].data<unsigned int>()=0;
  query->setCondition( "p1.tagid <> :tagid", bindData );
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
  coral::IQuery* leaftagquery=tagInventorytable.newQuery();
  leaftagquery->addToTableList( cond::TagDBNames::tagInventoryTable() );
  leaftagquery->addToOutputList( "pfn" );
  leaftagquery->addToOutputList( "recordname" );
  leaftagquery->addToOutputList( "labelname" );
  leaftagquery->addToOutputList( "timetype" );
  coral::AttributeList myresult;
  myresult.extend("pfn",typeid(std::string));
  myresult.extend("recordname",typeid(std::string));
  myresult.extend("labelname",typeid(std::string));
  myresult.extend("timetype",typeid(std::string));
  leaftagquery->defineOutput( myresult );
  coral::AttributeList bindVariableList;
  bindVariableList.extend("tagid",typeid(unsigned int));
  leaftagquery->setCondition( "tagid = :tagid",bindVariableList );
  leaftagquery->limitReturnedRows(1,0);
  for( it=itBeg; it!=itEnd; ++it ){
    cond::TagMetadata tagmetadata;
    bindVariableList["tagid"].data<unsigned int>()=*it;
    coral::ICursor& cursor2 =leaftagquery->execute();
    cursor2.next();
    const coral::AttributeList& row = cursor2.currentRow();
    std::string tagname=row["tagname"].data<std::string>();
    tagmetadata.pfn=row["pfn"].data<std::string>();
    tagmetadata.recordname=row["recordname"].data<std::string>();
    tagmetadata.labelname=row["labelname"].data<std::string>();
    tagmetadata.timetype=row["timetype"].data<std::string>();
    cursor2.close();
    result.insert( std::make_pair<std::string, cond::TagMetadata>(tagname,tagmetadata) );
  }
}

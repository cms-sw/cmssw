#include "CondCore/DBCommon/interface/DbConnection.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/TagMetadata.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/TableDescription.h"
#include "RelationalAccess/ITableDataEditor.h"
#include "RelationalAccess/ITablePrivilegeManager.h"
#include "RelationalAccess/IBulkOperation.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/AttributeSpecification.h"
#include <iostream>
int main(){
  const std::string tagTreeTable("TAGTREE_TABLE_MYTREE1");
  const std::string tagInventoryTable("TAGINVENTORY_TABLE");
  try{
    cond::DbConnection connection;
    connection.configuration().setMessageLevel( coral::Error );
    connection.configure();
    cond::DbSession session = connection.createSession();
    session.open( "sqlite_file:tagDB.db" );
    session.transaction().start(false);
    coral::TableDescription tagTreeTableDesc;
    coral::TableDescription tagInventoryTableDesc;
    tagTreeTableDesc.setName(tagTreeTable);
    tagTreeTableDesc.insertColumn("nodeid", coral::AttributeSpecification::typeNameForId( typeid(unsigned int) ) );
    tagTreeTableDesc.insertColumn("nodelabel", coral::AttributeSpecification::typeNameForId( typeid(std::string) ) );
    tagTreeTableDesc.insertColumn("lft", coral::AttributeSpecification::typeNameForId( typeid(unsigned int) ) );
    tagTreeTableDesc.insertColumn("rgt", coral::AttributeSpecification::typeNameForId( typeid(unsigned int) ) );
    tagTreeTableDesc.insertColumn("parentid", coral::AttributeSpecification::typeNameForId( typeid(unsigned int) ) );
    tagTreeTableDesc.insertColumn("tagid", coral::AttributeSpecification::typeNameForId( typeid(unsigned int) ) );
    tagTreeTableDesc.insertColumn("globalsince", coral::AttributeSpecification::typeNameForId( typeid(unsigned long long) ) );
    tagTreeTableDesc.insertColumn("globaltill", coral::AttributeSpecification::typeNameForId( typeid(unsigned long long) ) );
    tagTreeTableDesc.insertColumn("comment", coral::AttributeSpecification::typeNameForId( typeid(std::string) ) );
    std::vector<std::string> cols;
    cols.push_back( "nodeid" );
    tagTreeTableDesc.setPrimaryKey( cols );
    tagTreeTableDesc.setNotNullConstraint( "nodelabel" );
    tagTreeTableDesc.setNotNullConstraint( "lft" );
    tagTreeTableDesc.setNotNullConstraint( "rgt" );
    tagTreeTableDesc.setNotNullConstraint( "tagid" );
    tagTreeTableDesc.setUniqueConstraint( "nodelabel" );
    coral::ITable& treetable=session.nominalSchema().createTable( tagTreeTableDesc );
    treetable.privilegeManager().grantToPublic( coral::ITablePrivilegeManager::Select);
    
    tagInventoryTableDesc.setName(tagInventoryTable);
    tagInventoryTableDesc.insertColumn("tagid", coral::AttributeSpecification::typeNameForId( typeid(unsigned int) ) );
    tagInventoryTableDesc.insertColumn("tagname", coral::AttributeSpecification::typeNameForId( typeid(std::string) ) );
    tagInventoryTableDesc.insertColumn("pfn", coral::AttributeSpecification::typeNameForId( typeid(std::string) ) );
    tagInventoryTableDesc.insertColumn("recordname", coral::AttributeSpecification::typeNameForId( typeid(std::string) ) );
    tagInventoryTableDesc.insertColumn("objectname", coral::AttributeSpecification::typeNameForId( typeid(std::string) ) );
    tagInventoryTableDesc.insertColumn("labelname", coral::AttributeSpecification::typeNameForId( typeid(std::string) ) );
    tagInventoryTableDesc.insertColumn("comment", coral::AttributeSpecification::typeNameForId( typeid(std::string) ) );
    std::vector<std::string> pkcols;
    pkcols.push_back( "tagid" );
    tagInventoryTableDesc.setPrimaryKey( pkcols );
    tagInventoryTableDesc.setNotNullConstraint( "tagname" );
    tagInventoryTableDesc.setNotNullConstraint( "pfn" );
    tagInventoryTableDesc.setNotNullConstraint( "recordname" );
    tagInventoryTableDesc.setNotNullConstraint( "objectname" );
    tagInventoryTableDesc.setNotNullConstraint( "labelname" );
    tagInventoryTableDesc.setUniqueConstraint( "tagname" );
    coral::ITable& inventorytable=session.nominalSchema().createTable( tagInventoryTableDesc );
    inventorytable.privilegeManager().grantToPublic( coral::ITablePrivilegeManager::Select);
    std::cout<<"all tables are created"<<std::endl;
    std::cout<<"building tag collection"<<std::endl;
    std::vector<std::pair<std::string, cond::TagMetadata> > tagcollection;
    std::string tagname;
    cond::TagMetadata tagmetadata;
    tagname="mytest";
    tagmetadata.pfn="sqlite_file:mytest.db";
    tagmetadata.recordname="PedestalsRcd";
    tagmetadata.objectname="Pedestals";
    tagmetadata.labelname="lab3d";
    tagcollection.push_back(std::make_pair<std::string,cond::TagMetadata>(tagname,tagmetadata));
    tagname="mypedestals";
    tagmetadata.pfn="sqlite_file:mytest.db";
    tagmetadata.recordname="PedestalsRcd";
    tagmetadata.objectname="Pedestals";
    tagmetadata.labelname="lab2";
    tagcollection.push_back(std::make_pair<std::string,cond::TagMetadata>(tagname,tagmetadata));
    tagname="anothermytest";
    tagmetadata.pfn="sqlite_file:mytest.db";
    tagmetadata.recordname="anotherPedestalsRcd";
    tagmetadata.objectname="Pedestals";
    tagmetadata.labelname="";
    tagcollection.push_back(std::make_pair<std::string,cond::TagMetadata>(tagname,tagmetadata));

    std::cout<<"populate tag inventory"<<std::endl;
    coral::AttributeList data;
    std::cout << "Inserting new rows into the table." << std::endl;
    inventorytable.dataEditor().rowBuffer(data);
    coral::IBulkOperation* rowInserter = inventorytable.dataEditor().bulkInsert(data,5);
    std::vector< std::pair<std::string, cond::TagMetadata> >::iterator itBeg=tagcollection.begin();
    std::vector< std::pair<std::string, cond::TagMetadata> >::iterator itEnd=tagcollection.end();
    unsigned int tagid=0;
    for(std::vector< std::pair<std::string, cond::TagMetadata> >::iterator it=itBeg;it!=itEnd;++it) {
      ++tagid;
      data["tagid"].data<unsigned int>() = tagid;
      data["tagname"].data<std::string>() = it->first;
      data["pfn"].data<std::string>() = it->second.pfn;
      data["recordname"].data<std::string>() = it->second.recordname;
      data["objectname"].data<std::string>() = it->second.objectname;
      data["labelname"].data<std::string>() = it->second.labelname;
      rowInserter->processNextIteration();
    }
    rowInserter->flush();
    delete rowInserter;
    struct nodedata{
      unsigned int nodeid;
      std::string nodelabel;
      unsigned int lft;
      unsigned int rgt;
      unsigned int tagid;
    };
    std::cout<<"building test tag tree"<<std::endl;
    /*
                 All(1,12)
                /          \
      Calibration(2,7)     Alignment(8,11)
        /           \                /
      mytest(3,4) mypedestals(5,6) anothermytest(9,10)
    */
    coral::AttributeList nodedata;
    treetable.dataEditor().rowBuffer(nodedata);
    coral::IBulkOperation* treeInserter = treetable.dataEditor().bulkInsert(nodedata,6);    
    nodedata["nodeid"].data<unsigned int>()=1;
    nodedata["nodelabel"].data<std::string>()="All";
    nodedata["lft"].data<unsigned int>()=1;
    nodedata["rgt"].data<unsigned int>()=12;
    nodedata["parentid"].data<unsigned int>()=0;
    nodedata["globalsince"].data<unsigned long long>()=0;
    nodedata["globaltill"].data<unsigned long long>()=0;
    nodedata["comment"].data<std::string>()="";
    nodedata["tagid"].data<unsigned int>()=0;
    treeInserter->processNextIteration();

    nodedata["nodeid"].data<unsigned int>()=2;
    nodedata["nodelabel"].data<std::string>()="Calibration";
    nodedata["lft"].data<unsigned int>()=2;
    nodedata["rgt"].data<unsigned int>()=7;
    nodedata["parentid"].data<unsigned int>()=1;
    nodedata["globalsince"].data<unsigned long long>()=0;
    nodedata["globaltill"].data<unsigned long long>()=0;
    nodedata["comment"].data<std::string>()="";
    nodedata["tagid"].data<unsigned int>()=0;
    treeInserter->processNextIteration();

    nodedata["nodeid"].data<unsigned int>()=3;
    nodedata["nodelabel"].data<std::string>()="Alignment";
    nodedata["lft"].data<unsigned int>()=8;
    nodedata["rgt"].data<unsigned int>()=11;
    nodedata["parentid"].data<unsigned int>()=1;
    nodedata["globalsince"].data<unsigned long long>()=0;
    nodedata["globaltill"].data<unsigned long long>()=0;
    nodedata["comment"].data<std::string>()="";
    nodedata["tagid"].data<unsigned int>()=0;
    treeInserter->processNextIteration();

    nodedata["nodeid"].data<unsigned int>()=4;
    nodedata["nodelabel"].data<std::string>()="mycalib11";
    nodedata["lft"].data<unsigned int>()=3;
    nodedata["rgt"].data<unsigned int>()=4;
    nodedata["parentid"].data<unsigned int>()=2;
    nodedata["globalsince"].data<unsigned long long>()=0;
    nodedata["globaltill"].data<unsigned long long>()=0;
    nodedata["comment"].data<std::string>()="";
    nodedata["tagid"].data<unsigned int>()=1;
    treeInserter->processNextIteration();

    nodedata["nodeid"].data<unsigned int>()=5;
    nodedata["nodelabel"].data<std::string>()="mycalib2";
    nodedata["lft"].data<unsigned int>()=5;
    nodedata["rgt"].data<unsigned int>()=6;
    nodedata["parentid"].data<unsigned int>()=2;
    nodedata["globalsince"].data<unsigned long long>()=0;
    nodedata["globaltill"].data<unsigned long long>()=0;
    nodedata["comment"].data<std::string>()="";
    nodedata["tagid"].data<unsigned int>()=2;
    treeInserter->processNextIteration();
    
    nodedata["nodeid"].data<unsigned int>()=6;
    nodedata["nodelabel"].data<std::string>()="myalign1";
    nodedata["lft"].data<unsigned int>()=9;
    nodedata["rgt"].data<unsigned int>()=10;
    nodedata["parentid"].data<unsigned int>()=3;
    nodedata["globalsince"].data<unsigned long long>()=0;
    nodedata["globaltill"].data<unsigned long long>()=0;
    nodedata["comment"].data<std::string>()="";
    nodedata["tagid"].data<unsigned int>()=3;
    treeInserter->processNextIteration();
    treeInserter->flush();
    session.transaction().commit();
    delete treeInserter;
  }catch(const cond::Exception& er){
    std::cout<<"error "<<er.what()<<std::endl;
  }catch(const std::exception& er){
    std::cout<<"std error "<<er.what()<<std::endl;
  }
}

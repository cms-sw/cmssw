#include "CondCore/ORA/interface/Exception.h"
#include "MappingToSchema.h"
#include "MappingTree.h"
#include "MappingRules.h"
//
// externals
#include "CoralBase/Blob.h"
#include "CoralBase/AttributeSpecification.h"
#include "RelationalAccess/IColumn.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/ITablePrivilegeManager.h"
#include "RelationalAccess/SchemaException.h"
#include "RelationalAccess/TableDescription.h"

ora::MappingToSchema::MappingToSchema( coral::ISchema& schema ):
  m_schema( schema ){
}

ora::MappingToSchema::~MappingToSchema(){
}

void ora::MappingToSchema::createTable( const TableInfo& tableInfo ){
  coral::TableDescription description("ORA");
  description.setName(tableInfo.m_tableName);
  std::vector<std::string> columnsForIndex;
  std::vector<std::string> columnsForFk;
  size_t i=0;
  size_t cols = tableInfo.m_idColumns.size();
  for( std::vector<std::string>::const_iterator iCol = tableInfo.m_idColumns.begin();
       iCol != tableInfo.m_idColumns.end(); ++iCol ){
    description.insertColumn( *iCol, coral::AttributeSpecification::typeNameForId( typeid(int) ) );
    description.setNotNullConstraint( *iCol );
    if( !tableInfo.m_dependency ) {
      columnsForIndex.push_back( *iCol );
      if( i< cols-1 ) columnsForFk.push_back( *iCol );
    } else {
      if( i>0 ) columnsForIndex.push_back( *iCol );
      if( i>0 && i< cols-1 ) columnsForFk.push_back( *iCol );
    }
    ++i;
  }
  for( std::map<std::string,std::string>::const_iterator iDataCol = tableInfo.m_dataColumns.begin();
       iDataCol != tableInfo.m_dataColumns.end(); ++iDataCol ){
    description.insertColumn( iDataCol->first, iDataCol->second );
    description.setNotNullConstraint( iDataCol->first );
  }
  description.setPrimaryKey( columnsForIndex );
  if( !tableInfo.m_parentTableName.empty() ){
    std::string fkName = MappingRules::fkNameForIdentity( tableInfo.m_tableName );
    
    if( !tableInfo.m_dependency ) {
      description.createForeignKey( fkName, columnsForFk, tableInfo.m_parentTableName, tableInfo.m_refColumns );
    } else {
      std::vector<std::string> refCols;
      for(size_t i=1;i<tableInfo.m_refColumns.size();i++) refCols.push_back( tableInfo.m_refColumns[i] );
      if( !refCols.empty() ) description.createForeignKey( fkName, columnsForFk, tableInfo.m_parentTableName, refCols );
    }
  }
  m_schema.createTable( description );
  //.privilegeManager().grantToPublic( coral::ITablePrivilegeManager::Select );
  
}

void ora::MappingToSchema::create( const MappingTree& mapping ){
  std::vector<TableInfo> tableList = mapping.tables();
  for( std::vector<TableInfo>::iterator iT = tableList.begin();
       iT != tableList.end(); ++iT ){
    createTable( *iT );
  }
  
}

void ora::MappingToSchema::alter( const MappingTree& mapping ){
  std::vector<TableInfo> tableList = mapping.tables();
  for( std::vector<TableInfo>::iterator iT = tableList.begin();
       iT != tableList.end(); ++iT ){
    if( m_schema.existsTable( iT->m_tableName ) ){
      std::set<std::string> allCols;
      coral::ITable& table = m_schema.tableHandle( iT->m_tableName );
      // check all of the columns
      for( std::vector<std::string>::const_iterator iCol = iT->m_idColumns.begin();
           iCol != iT->m_idColumns.end(); ++iCol ){
        try {
          table.description().columnDescription( *iCol );
        } catch ( const coral::InvalidColumnNameException&){
          // not recoverable: id columns cannot be added.
          throwException("ID Column \""+*iCol+"\" has not been found in table \""+iT->m_tableName+"\" as required in the mapping.",
                         "MappingToSchema::alter");
        }
        allCols.insert( *iCol );
      }
      for( std::map<std::string,std::string>::const_iterator iDataCol = iT->m_dataColumns.begin();
           iDataCol != iT->m_dataColumns.end(); ++iDataCol ){
        try {
          const coral::IColumn& colDescr = table.description().columnDescription( iDataCol->first );
          // check the type
          if( colDescr.type() != iDataCol->second ){
            // not recoverable: column type cannot be changed.
            throwException("ID Column \""+iDataCol->first+"\" in table \""+iT->m_tableName+"\" is type \""+colDescr.type()+
                           "\" while is required of type \""+iDataCol->second+"\" in the mapping.",
                           "MappingToSchema::alter");
          }
          
        } catch ( const coral::InvalidColumnNameException&){
          table.schemaEditor().insertColumn( iDataCol->first, iDataCol->second );
          table.schemaEditor().setNotNullConstraint( iDataCol->first );
        }        
        allCols.insert( iDataCol->first );
      }
      // then check the unused columns for not null constraint
      int ncols = table.description().numberOfColumns();
      for( int i=0;i<ncols;i++ ){
        const coral::IColumn& colDescr = table.description().columnDescription( i );
        std::set<std::string>::const_iterator iC = allCols.find( colDescr.name() );
        if( iC == allCols.end() ){
          table.schemaEditor().setNotNullConstraint( colDescr.name(), false );
        }
      }
    } else {
      createTable( *iT );
    }
  }
}

bool ora::MappingToSchema::check( const MappingTree& mapping ){
  bool ok = true;
  std::vector<TableInfo> tableList = mapping.tables();
  for( std::vector<TableInfo>::iterator iT = tableList.begin();
       iT != tableList.end(); ++iT ){
    if( m_schema.existsTable( iT->m_tableName ) ){
      ok = false;
    }
  }
  return ok;
}




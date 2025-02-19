#include "CondCore/ORA/interface/Exception.h"
#include "TableRegister.h"
// externals
#include "RelationalAccess/IColumn.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/ITableDescription.h"

ora::TableRegister::TableRegister( coral::ISchema& schema ):
  m_schema( schema ),m_init(false),m_register(),m_currentTable(0),m_currentColumns(0){
}

ora::TableRegister::~TableRegister(){
}

void ora::TableRegister::init(){
  if(!m_init){
    std::set<std::string> tableList = m_schema.listTables();
    for( std::set<std::string>::const_iterator iT = tableList.begin(); iT != tableList.end(); ++iT ){
      coral::ITable& table = m_schema.tableHandle( *iT );
      std::map<std::string,std::set<std::string> >::iterator iEntry =
        m_register.insert( std::make_pair( *iT, std::set<std::string>() ) ).first;
      int ncols = table.description().numberOfColumns();
      for(int i=0;i<ncols;i++){
        iEntry->second.insert( table.description().columnDescription( i ).name() );
      }
    }
    m_init = true;
  }
}

bool ora::TableRegister::checkTable(const std::string& tableName){
  init();
  if(!m_currentTable || (tableName!=*m_currentTable)){
    std::map<std::string, std::set<std::string> >::iterator iT = m_register.find( tableName );
    if(iT==m_register.end()) {
      m_currentTable = 0;
      m_currentColumns = 0;
      return false;
    }
    m_currentTable = &iT->first;
    m_currentColumns = &iT->second;
  }
  return true;
}

bool ora::TableRegister::checkColumn(const std::string& tableName,
                                     const std::string& columnName){
  init();
  if(!m_currentTable || (tableName!=*m_currentTable)){
    std::map<std::string, std::set<std::string> >::iterator iT = m_register.find( tableName );
    if(iT==m_register.end()) {
      m_currentTable = 0;
      m_currentColumns = 0;      
      return false;
    }
    m_currentTable = &iT->first;
    m_currentColumns = &iT->second;
  }
  return m_currentColumns->find(columnName)!=m_currentColumns->end();
}

void
ora::TableRegister::insertTable(const std::string& tableName){
  init();
  m_register.insert( std::make_pair(tableName, std::set<std::string>()) );
  m_currentTable = 0;
  m_currentColumns = 0;
}

bool ora::TableRegister::insertColumn(const std::string& tableName,
                                      const std::string& columnName){
  init();
  if(!m_currentTable || (tableName!=*m_currentTable)){
    std::map<std::string, std::set<std::string> >::iterator iT = m_register.find( tableName );
    if(iT==m_register.end()) {
      m_currentTable = 0;
      m_currentColumns = 0;      
      return false;
    }
    m_currentTable = &iT->first;
    m_currentColumns = &iT->second;
  }
  m_currentColumns->insert( columnName );
  return true;
}

bool ora::TableRegister::insertColumns(const std::string& tableName,
                                       const std::vector<std::string>& columns ){
  
  init();
  if(!m_currentTable || (tableName!=*m_currentTable)){
    std::map<std::string, std::set<std::string> >::iterator iT = m_register.find( tableName );
    if(iT==m_register.end()) {
      m_currentTable = 0;
      m_currentColumns = 0;      
      return false;
    }
    m_currentTable = &iT->first;
    m_currentColumns = &iT->second;
  }
  for( std::vector<std::string>::const_iterator iC = columns.begin(); iC != columns.end(); iC++ ){
    m_currentColumns->insert( *iC );
  }
  return true;
}

size_t ora::TableRegister::numberOfColumns(const std::string& tableName){
  init();
  if(!m_currentTable || (tableName!=*m_currentTable)){
    std::map<std::string, std::set<std::string> >::iterator iT = m_register.find( tableName );
    if(iT==m_register.end()) {
      m_currentTable = 0;
      m_currentColumns = 0;      
      return 0;
    }
    m_currentTable = &iT->first;
    m_currentColumns = &iT->second;
  }
  return m_currentColumns->size();
}



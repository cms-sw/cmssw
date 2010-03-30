#include "TableRegister.h"

ora::TableRegister::TableRegister():m_register(),m_currentTable(0),m_currentColumns(0){
}

ora::TableRegister::~TableRegister(){
}

bool ora::TableRegister::checkTable(const std::string& tableName){
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
  m_register.insert( std::make_pair(tableName, std::set<std::string>()) );
  m_currentTable = 0;
  m_currentColumns = 0;
}

bool ora::TableRegister::insertColumn(const std::string& tableName,
                                      const std::string& columnName){
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

size_t ora::TableRegister::numberOfColumns(const std::string& tableName){
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


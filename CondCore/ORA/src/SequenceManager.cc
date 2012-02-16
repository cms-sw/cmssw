#include "CondCore/ORA/interface/SequenceManager.h"
#include "OraDatabaseSchema.h"
#include "Sequences.h"

ora::SequenceManager::SequenceManager( const std::string& tableName, 
				       coral::ISchema& schema ):
  m_table(),
  m_impl(){
  m_table.reset( new OraSequenceTable( tableName, schema ) );
  m_impl.reset( new Sequences( *m_table ) );
}

ora::SequenceManager::SequenceManager( const SequenceManager& rhs ):
  m_table( rhs.m_table ),
  m_impl( rhs.m_impl ){
}
    
ora::SequenceManager::~SequenceManager(){
}

ora::SequenceManager& ora::SequenceManager::operator=( const SequenceManager& rhs ){
  if( this != &rhs ){
    m_table = rhs.m_table;
    m_impl = rhs.m_impl;
  }
  return *this;
}

std::string ora::SequenceManager::tableName(){
  return m_table->name();
}

void ora::SequenceManager::create( const std::string& sequenceName ){
  if( !m_table->exists() ){
    m_table->create();
  }
  m_impl->create( sequenceName );
}

int ora::SequenceManager::getNextId( const std::string& sequenceName, 
				     bool sinchronize ){
  return m_impl->getNextId( sequenceName, sinchronize );
}

void ora::SequenceManager::sinchronize( const std::string& sequenceName ){
  m_impl->sinchronize( sequenceName );
}

void ora::SequenceManager::sinchronizeAll(){
  m_impl->sinchronizeAll();
}

void ora::SequenceManager::erase( const std::string& sequenceName ){
  m_impl->erase( sequenceName );
}

void ora::SequenceManager::clear(){
  m_impl->clear();
}

#include "CondCore/ORA/interface/Exception.h"
#include "Sequences.h"
#include "IDatabaseSchema.h"

ora::Sequences::Sequences( ora::IDatabaseSchema& schema ):
  m_lastIds(),
  m_table( schema.sequenceTable()){
}

ora::Sequences::Sequences( ora::ISequenceTable& table ):
  m_lastIds(),
  m_table( table ){
}

ora::Sequences::~Sequences(){
}

void ora::Sequences::create( const std::string& sequenceName ){
  m_table.add( sequenceName );
}

int ora::Sequences::getNextId( const std::string& sequenceName, bool sinchronize ){
  int next = 0;
  std::map<std::string,int>::iterator iS = m_lastIds.find( sequenceName );
  if( iS == m_lastIds.end() ){
    bool found = m_table.getLastId( sequenceName, next );
    if( ! found ) {
      throwException("Sequence \""+sequenceName+"\" does not exists.","Sequences::getNextId");
    } else {
      next += 1;
    }
    m_lastIds.insert( std::make_pair( sequenceName, next ));
  } else {
    next = ++iS->second;
  }

  if( sinchronize){
    m_table.sinchronize( sequenceName, next );
  }  
  return next;
}

void ora::Sequences::sinchronize( const std::string& sequenceName ){
  std::map<std::string,int>::iterator iS = m_lastIds.find( sequenceName );
  if( iS != m_lastIds.end() ){
    int lastOnDb = 0;
    m_table.getLastId( sequenceName, lastOnDb );
    if( lastOnDb < iS->second ) m_table.sinchronize( sequenceName, iS->second );
    m_lastIds.erase( sequenceName );
  }
}

void ora::Sequences::sinchronizeAll(){
  for( std::map<std::string,int>::iterator iS = m_lastIds.begin();
       iS != m_lastIds.end(); iS++ ){
    int lastOnDb = 0;
    m_table.getLastId( iS->first, lastOnDb );
    if( lastOnDb < iS->second ) m_table.sinchronize( iS->first, iS->second );    
  }
  clear();
}

void ora::Sequences::erase( const std::string& sequenceName ){
  m_table.erase( sequenceName );
}

void ora::Sequences::clear(){
  m_lastIds.clear();
}

ora::NamedSequence::NamedSequence( const std::string& sequenceName, ora::IDatabaseSchema& dbSchema ):
  m_name( sequenceName ),
  m_sequences( dbSchema ){
}

ora::NamedSequence::~NamedSequence(){
}

void ora::NamedSequence::create(){
  m_sequences.create( m_name );
}

int ora::NamedSequence::getNextId( bool sinchronize ){
  return m_sequences.getNextId( m_name, sinchronize );
}

void ora::NamedSequence::sinchronize(){
  m_sequences.sinchronize( m_name );
}

void ora::NamedSequence::erase(){
  m_sequences.erase( m_name );
}

void ora::NamedSequence::clear(){
  m_sequences.clear();
}


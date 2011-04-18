#include "CondCore/ORA/interface/OId.h"
//
#include <cstdio>
#include <cstring>

static const char* OIDFMT = "%04X-%08X";
static const size_t OIDSIZ = 13;

bool ora::OId::isOId( const std::string& input ){
  ora::OId tmp;
  return tmp.fromString( input );
}

ora::OId::OId():
  m_containerId(-1),
  m_itemId(-1){
}

ora::OId::OId( const std::pair<int,int>& oidPair ):
  m_containerId( oidPair.first ),
  m_itemId( oidPair.second ){
}
  
ora::OId::OId( int contId, int itemId ):
  m_containerId( contId),
  m_itemId( itemId ){
}

ora::OId::OId( const OId& rhs ):
  m_containerId( rhs.m_containerId),
  m_itemId( rhs.m_itemId ){
}

ora::OId& ora::OId::operator=( const OId& rhs ){
  m_containerId = rhs.m_containerId;
  m_itemId = rhs.m_itemId;
  return *this;
}

bool ora::OId::operator==( const OId& rhs ) const {
  if(m_containerId != rhs.m_containerId ) return false;
  if(m_itemId != rhs.m_itemId ) return false;
  return true;
}

bool ora::OId::operator!=( const OId& rhs ) const {
  return !operator==(rhs);
}

int ora::OId::containerId() const{
  return m_containerId;
}

int ora::OId::itemId() const{
  return m_itemId;
}

std::string ora::OId::toString() const {
  char text[OIDSIZ];
  ::sprintf(text, OIDFMT, m_containerId, m_itemId );
  return std::string(text);
}

bool ora::OId::fromString( const std::string& source ){
  if(source.size()>OIDSIZ) return false; // constraint relaxed...
  const char* ptr = source.c_str();
  if( ::sscanf( ptr, OIDFMT, &m_containerId, &m_itemId )==2 ) return true;
  return false;
}

void ora::OId::toOutputStream( std::ostream& os ) const {
  os << this->toString();
}

void ora::OId::reset() {
  m_containerId = -1;
  m_itemId = -1;
}

bool ora::OId::isInvalid() const {
  return (m_containerId == -1 || m_itemId == -1);
}

std::pair<int,int> ora::OId::toPair() const {
  return std::make_pair( m_containerId, m_itemId );
}

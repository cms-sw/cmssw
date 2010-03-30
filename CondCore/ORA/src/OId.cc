#include "CondCore/ORA/interface/OId.h"
//
#include <cstdio>
#include <cstring>

static const char* fmtContId = "[CID=%08X]";
static const char* fmtItemId = "[OID=%08X]";

ora::OId::OId():
  m_containerId(-1),
  m_itemId(-1){
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

bool ora::OId::operator==( const OId& rhs ){
  if(m_containerId != rhs.m_containerId ) return false;
  if(m_itemId != rhs.m_itemId ) return false;
  return true;
}

bool ora::OId::operator!=( const OId& rhs ){
  return !operator==(rhs);
}

int ora::OId::containerId() const{
  return m_containerId;
}

int ora::OId::itemId() const{
  return m_itemId;
}

std::string ora::OId::toString(){
  std::string str("");
  char text[128];
  ::sprintf(text, fmtContId, m_containerId);
  str += text;
  ::sprintf(text, fmtItemId, m_itemId);
  str += text;
  return str;
}

void ora::OId::fromString( const std::string& source ){
  std::string tmp = source;
  for(char* p1 = (char*)tmp.c_str(); p1; p1 = ::strchr(++p1,'[')) {
    char* p2 = ::strchr(p1, '=');
    char* p3 = ::strchr(p1, ']');
    if ( p2 && p3 )   {
      if ( ::strncmp(fmtContId, p1, 4) == 0 )  {
        ::sscanf(p1, fmtContId, &m_containerId );
      }
      else if ( ::strncmp(fmtItemId, p1, 4) == 0 )  {
        ::sscanf(p1, fmtItemId, &m_itemId );
      }
      else    {
        *p3 = *p2 = 0;
      }
      *p3 = ']';
      *p2 = '=';
    }
  }
}


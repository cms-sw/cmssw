#include "CondCore/ORA/interface/Version.h"
//
#include <cstdio>
#include <cstring>

static const char* thisSchemaVersionLabel = "1.1.0";

static const char* poolSchemaVersionLabel = "POOL";
static const char* fmt = "%3d.%3d.%3d";

ora::Version& ora::Version::poolSchemaVersion(){
  static Version s_ver;
  s_ver.m_label = std::string(poolSchemaVersionLabel);
  s_ver.m_main = -1;
  return s_ver;
}

ora::Version& ora::Version::thisSchemaVersion(){
  static Version s_ver;
  s_ver = fromString( std::string( thisSchemaVersionLabel ));
  return s_ver;
}

ora::Version ora::Version::fromString( const std::string& source ){
  if ( source == poolSchemaVersionLabel ) return poolSchemaVersion();
  Version ver;
  ver.m_label = source;
  std::string tmp = source;
  char* p = (char*)tmp.c_str();
  ::sscanf(p, fmt, &ver.m_main, &ver.m_release, &ver.m_patch );
  return ver;
}

ora::Version::Version():
  m_label(""),
  m_main( -999 ),
  m_release( 0 ),
  m_patch( 0 ){
}

ora::Version::Version( const Version& rhs ):
  m_label( rhs.m_label ),
  m_main( rhs.m_main ),
  m_release( rhs.m_release ),
  m_patch( rhs.m_patch ){
}
  
ora::Version& ora::Version::operator=( const Version& rhs ){
  m_label = rhs.m_label;
  m_main = rhs.m_main;
  m_release = rhs.m_release;
  m_patch = rhs.m_patch;
  return *this;
}

bool ora::Version::operator==( const Version& rhs ) const {
  return m_main == rhs.m_main && m_release == rhs.m_release && m_patch == rhs.m_patch;
}

bool ora::Version::operator!=( const Version& rhs ) const{
  return !operator==( rhs );
}

bool ora::Version::operator>( const Version& rhs ) const {
  if( m_main > rhs.m_main ) return true;
  if( m_main == rhs.m_main ){
    if( m_release > rhs.m_release ) return true;
    if( m_release == rhs.m_release ){
      if(m_patch > rhs.m_patch ) return true;
    }
  }
  return false;
}

bool ora::Version::operator<( const Version& rhs ) const {
  if( m_main < rhs.m_main ) return true;
  if( m_main == rhs.m_main ){
    if( m_release < rhs.m_release ) return true;
    if( m_release == rhs.m_release ){
      if(m_patch < rhs.m_patch ) return true;
    }
  }
  return false;
}

bool ora::Version::operator>=( const Version& rhs ) const {
  if( m_main >= rhs.m_main ) return true;
  if( m_main == rhs.m_main ){
    if( m_release >= rhs.m_release ) return true;
    if( m_release == rhs.m_release ){
      if(m_patch >= rhs.m_patch ) return true;
    }
  }
  return false;
}

bool ora::Version::operator<=( const Version& rhs ) const {
  if( m_main <= rhs.m_main ) return true;
  if( m_main == rhs.m_main ){
    if( m_release <= rhs.m_release ) return true;
    if( m_release == rhs.m_release ){
      if(m_patch <= rhs.m_patch ) return true;
    }
  }
  return false;
}

std::string ora::Version::toString() const {
  return m_label;
}

void ora::Version::toOutputStream( std::ostream& os ) const {
  os << m_label;
}


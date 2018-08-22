#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/Singleton.h"

#include <ext/alloc_traits.h>
#include <cstdlib>
#include <sstream>

std::ostream & operator<<( std::ostream & os, const DDName & n )
{ 
  os << n.ns() << ':' << n.name();
  return os;
}  

DDName::DDName( const std::string & name, const std::string & ns )
 : id_( registerName( std::make_pair( name, ns ))->second )
{ }

DDName::DDName( const std::string & name )
 : id_( 0 )
{ 
  std::pair<std::string, std::string> result = DDSplit( name );
  if( result.second.empty()) {
    id_ = registerName( std::make_pair( result.first, DDCurrentNamespace::ns()))->second;
  }  
  else {
    id_ = registerName( result )->second;
  }
}

DDName::DDName( const char* name )
 : id_( 0 )
{ 
  std::pair< std::string, std::string > result = DDSplit( name );
  if( result.second.empty()) {
    id_ = registerName( std::make_pair( result.first, DDCurrentNamespace::ns()))->second;
  }  
  else {
    id_ = registerName( result )->second;
  }
}

DDName::DDName( const char* name, const char* ns )
 : id_( registerName( std::make_pair( std::string( name ), std::string( ns )))->second )
{ }

DDName::DDName()
  : id_(0)
{ }

const std::string &
DDName::name() const 
{
  const static std::string ano_( "anonymous" );
  const std::string * result;
  if( id_ < 0 ) {
      result = &ano_;
  }
  else {
    result = &DDI::Singleton<IdToName>::instance()[id_]->first.first;
  }
  return *result; 
}

const std::string &
DDName::ns() const
{
  const static std::string ano_( "anonymous" );
  const std::string * result;
  if( id_ < 0 ) {
    result = &ano_;
  }
  else {
    result = &DDI::Singleton<IdToName>::instance()[id_]->first.second;
  }
  return *result;
}

DDName::Registry::iterator
DDName::registerName( const std::pair<std::string, std::string> & nm )
{
  Registry& reg = DDI::Singleton<Registry>::instance();
  IdToName & idToName = DDI::Singleton<IdToName>::instance();  
  Registry::size_type sz = reg.size();
  if( !sz )
  {
    reg[std::make_pair( std::string(""), std::string(""))] = 0;
    idToName.emplace_back( reg.begin());
    ++sz;
  }
  Registry::value_type val( nm, sz );
  std::pair<Registry::iterator, bool> result = reg.insert( val );
  if( result.second ) {
    idToName.emplace_back( result.first );
  }
  return result.first;
}

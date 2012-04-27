#include "DetectorDescription/Core/interface/DDValue.h"
#include "DetectorDescription/Base/interface/DDException.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <cassert>

void
DDValue::init( const std::string &name )
{
  unsigned int temp = indexer().size()+1;
  typedef std::map<std::string,unsigned int>::iterator itT;
  std::pair<itT,bool> result = indexer().insert( std::make_pair( name, temp ));
  
  if( result.second )
  {
    id_ = temp;
    names().push_back( name );
  }
  else
  {
    id_ = result.first->second;
  }  
}

DDValue::DDValue( const std::string & name )
 : id_( 0 ),
   vecPair_( 0 )
{
  init( name );
}

DDValue::DDValue( const char * name )
 : id_( 0 ),
   vecPair_( 0 )
{
  init( name );
}

DDValue::DDValue( const std::string & name, const std::vector<DDValuePair>& v ) 
 : id_( 0 )
{
  init( name );
  
  std::vector<DDValuePair>::const_iterator it = v.begin();
  std::vector<std::string> svec;
  std::vector<double> dvec;
  vecPair_ = new vecpair_type( false, std::make_pair( svec, dvec ));
  mem( vecPair_ );
  for(; it != v.end(); ++it )
  {
    vecPair_->second.first.push_back( it->first );
    vecPair_->second.second.push_back( it->second );
  }
}


DDValue::DDValue( const std::string & name, double val ) 
 : id_( 0 )
{
  init( name );
  
  std::vector<std::string> svec( 1, "" ); 
  std::vector<double> dvec( 1, val );
  
  vecPair_ =  new vecpair_type( false, std::make_pair( svec, dvec ));
  setEvalState( true );
  mem( vecPair_ );
}

DDValue::DDValue( const std::string & name, const std::string & sval, double dval ) 
 : id_( 0 )
{
  init( name );
  
  std::vector<std::string> svec( 1, sval );
  std::vector<double> dvec( 1, dval );
  vecPair_ = new vecpair_type( false, std::make_pair( svec, dvec ));
  setEvalState( true );
  mem( vecPair_ );
}

DDValue::DDValue( const std::string & name, const std::string & sval ) 
 : id_( 0 )
{
  init( name );
  
  std::vector<std::string> svec( 1, sval );
  std::vector<double> dvec( 1, 0 );
  vecPair_ = new vecpair_type( false, std::make_pair( svec, dvec ));
  setEvalState( false );
  mem( vecPair_ );
}

DDValue::DDValue( unsigned int i ) 
 : id_( 0 ),
   vecPair_( 0 )
{
  if( names().size() - 1 <= i )
    id_ = i;
}

DDValue::~DDValue( void )
{}

void
DDValue::clear( void )
{
  std::vector<boost::shared_ptr<vecpair_type> > & v = mem( 0 );
  v.clear();
}

std::map<std::string, unsigned int>&
DDValue::indexer( void )
{ 
  static std::map<std::string,unsigned int> indexer_;
  return indexer_;
}  
  
std::vector<std::string>&
DDValue::names( void )
{
  static std::vector<std::string> names_( 1 );
  return names_;
} 

std::vector<boost::shared_ptr<DDValue::vecpair_type> >&
DDValue::mem( DDValue::vecpair_type * vp )
{
  static std::vector<boost::shared_ptr<vecpair_type> > memory_;
  memory_.push_back( boost::shared_ptr<vecpair_type>( vp ));
  return memory_;
}

const std::vector<double> &
DDValue::doubles( void ) const 
{ 
  if( vecPair_->first )
  {
    return vecPair_->second.second; 
  }
  else
  {
    std::string message = "DDValue " + names()[id_] + " is not numerically evaluated! Use DDValue::std::strings()!";
    edm::LogError("DDValue") << message << std::endl;
    throw DDException(message);
  }
}

std::ostream & operator<<( std::ostream & o, const DDValue & v )
{
  o << v.name() << " = ";
  unsigned int i = 0;
  if( v.isEvaluated())
  {
    for(; i < v.size(); ++i )
    {
      o << '(' << v[i].first << ',' << v[i].second << ") ";
    }  
  }
  else
  {
    const std::vector<std::string> & s = v.strings();
    for(; i < v.size(); ++i )
    {
      o << s[i] << ' ';
    }
  }  
  return o;
}

//FIXME move it elsewhere; DO NOT put out the name for now... need to fix DDCoreToDDXMLOutput
std::ostream & operator<<( std::ostream & o, const DDValuePair & v )
{
  return o << v.second;
}

DDValuePair
DDValue::operator[]( unsigned int i ) const
{ 
  if( vecPair_->first )
  {
    return DDValuePair( vecPair_->second.first[i], vecPair_->second.second[i] );
  }
  else
  {
    std::string message = "DDValue " + names()[id_] + " is not numerically evaluated! Use DDValue::std::strings()!";
    edm::LogError( "DDValue" ) << message;
    throw DDException( message );
  }
}

void
DDValue::setEvalState( bool newState )
{
  vecPair_->first = newState; 
}

bool
DDValue::isEvaluated( void ) const
{
  return vecPair_->first;
}

bool
DDValue::operator==( const DDValue & v ) const 
{
  bool result( false );
  if( id() == v.id())
  { 
    assert( vecPair_ );
    assert( v.vecPair_ );
    if( vecPair_->first ) { // numerical values
      result = ( vecPair_->second.second == v.vecPair_->second.second );
    }  
    else { // std::string values
      result = ( vecPair_->second.first == v.vecPair_->second.first );
    }
  }
  return result;
}

bool
DDValue::operator<( const DDValue & v ) const 
{
  bool result( false );
  if( id() < v.id())
  { 
    result = true;
  }
  else
  {
    if( id() == v.id())
    {
      assert( vecPair_ );
      assert( v.vecPair_ );
      if( vecPair_->first && v.vecPair_->first ) { // numerical values
        result = ( vecPair_->second.second < v.vecPair_->second.second );
      }  
      else { // std::string values
        result = ( vecPair_->second.first < v.vecPair_->second.first );
      }
    }
  }
  return result;
}

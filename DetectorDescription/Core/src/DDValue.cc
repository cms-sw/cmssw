#include "DetectorDescription/Core/interface/DDValue.h"

#include "DetectorDescription/Core/interface/DDPosData.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDComparator.h"

#include <cassert>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

static std::atomic<unsigned int> lastIndex{0};

void
DDValue::init( const std::string &name )
{
  auto result = indexer().insert( { name, 0 } );
  
  auto& indexToUse = result.first->second;
  
  //A 0 index means either
  // 1) this result was just added or
  // 2) another thread just added this but has not yet assigned an index
  if(0 == indexToUse.value_) {
    auto newIndex = ++lastIndex;
    unsigned int previous = 0;
    indexToUse.value_.compare_exchange_strong(previous,newIndex);
  }
  id_ = indexToUse.value_.load();
  
  //Make sure the name is recorded at the proper index
  auto& allNames = names();
  allNames.grow_to_at_least(id_+1);
  auto& storedName = allNames[id_];
  if(not storedName.string_) {
    std::unique_ptr<std::string> newName(new std::string{name});
    std::string* previous = nullptr;
    if( storedName.string_.compare_exchange_strong(previous,newName.get())) {
      newName.release();
    }
  }
}

DDValue::DDValue( const std::string & name )
 : id_( 0 ),
   vecPair_()
{
  init( name );
}

DDValue::DDValue( const char * name )
 : id_( 0 ),
   vecPair_()
{
  init( name );
}

DDValue::DDValue( const std::string & name, const std::vector<DDValuePair>& v ) 
 : id_( 0 )
{
  init( name );
  
  auto it = v.begin();
  std::vector<std::string> svec;
  std::vector<double> dvec;
  vecPair_.reset(new vecpair_type( false, std::make_pair( svec, dvec )));
  for(; it != v.end(); ++it )
  {
    vecPair_->second.first.emplace_back( it->first );
    vecPair_->second.second.emplace_back( it->second );
  }
}

DDValue::DDValue( const std::string & name, double val ) 
 : id_( 0 )
{
  init( name );
  
  std::vector<std::string> svec( 1, "" ); 
  std::vector<double> dvec( 1, val );
  
  vecPair_.reset( new vecpair_type( false, std::make_pair( svec, dvec )));
  setEvalState( true );
}

DDValue::DDValue( const std::string & name, const std::string & sval, double dval ) 
 : id_( 0 )
{
  init( name );
  
  std::vector<std::string> svec( 1, sval );
  std::vector<double> dvec( 1, dval );
  vecPair_.reset(new vecpair_type( false, std::make_pair( svec, dvec )));
  setEvalState( true );
}

DDValue::DDValue( const std::string & name, const std::string & sval ) 
 : id_( 0 )
{
  init( name );
  
  std::vector<std::string> svec( 1, sval );
  std::vector<double> dvec( 1, 0 );
  vecPair_.reset(new vecpair_type( false, std::make_pair( svec, dvec )));
  setEvalState( false );
}

DDValue::DDValue( unsigned int i ) 
 : id_( 0 ),
   vecPair_()
{
  if( lastIndex >= i )
    id_ = i;
}

DDValue::NamesToIndicies&
DDValue::indexer( void )
{ 
  static NamesToIndicies indexer_;
  return indexer_;
}  

DDValue::Names DDValue::initializeNames() {
  //Make sure memory is zeroed before allocating StringHolder
  // this allows us to check the value of the held std::atomic
  // as the object is being added to the container
  DDValue::Names names{};
  names.emplace_back(StringHolder(std::string{}));
  return names;
}

DDValue::Names&
DDValue::names( void )
{
  static Names names_{ initializeNames() };
  return names_;
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
    std::string message = "DDValue " + name() + " is not numerically evaluated! Use DDValue::std::strings()!";
    edm::LogError("DDValue") << message << std::endl;
    throw cms::Exception("DDException") << message;
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
    std::string message = "DDValue " + name() + " is not numerically evaluated! Use DDValue::std::strings()!";
    edm::LogError( "DDValue" ) << message;
    throw cms::Exception("DDException") << message;
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

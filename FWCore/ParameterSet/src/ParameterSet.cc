// ----------------------------------------------------------------------
// $Id: ParameterSet.cc,v 1.1 2005/05/29 02:29:55 wmtan Exp $
//
// definition of ParameterSet's function members
// ----------------------------------------------------------------------

// ----------------------------------------------------------------------
// prerequisite source files and headers
// ----------------------------------------------------------------------

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSet.icc"

#include "FWCore/ParameterSet/interface/split.h"
#include "FWCore/ParameterSet/interface/types.h"
#include <algorithm>
#include <utility>

// ----------------------------------------------------------------------
// class invariant checker
// ----------------------------------------------------------------------

namespace edm {

  void
  ParameterSet::validate() const {
  }  // ParameterSet::validate()
  
  
  // ----------------------------------------------------------------------
  // constructors
  // ----------------------------------------------------------------------
  
  // ----------------------------------------------------------------------
  // coded string
  
  ParameterSet::ParameterSet( std::string const& code ) : tbl() {
    if( ! fromString(code) )
      throw ParameterSetError( "bad encoded ParameterSet string " + code );
    validate();
  }
  
  // ----------------------------------------------------------------------
  // Entry-handling
  // ----------------------------------------------------------------------
  
  Entry const&
  ParameterSet::retrieve( std::string const& name ) const {
    table::const_iterator  it = tbl.find(name);
    if( it == tbl.end() ) {
      it = tbl.find("label");
      if( it == tbl.end() )
        throw ParameterSetError( "'" + name + "' is not known in this anonymous ParameterSet" );
      else
        throw ParameterSetError( "'" + name + "' is not known in ParameterSet '"
                        + it->second.getString() + "'");
    }
    return it->second;
  }  // retrieve()
  
  Entry const* const
  ParameterSet::retrieveUntracked( std::string const& name ) const {
    table::const_iterator  it = tbl.find(name);
    if( it == tbl.end() )  {
      return 0;
    }
    return &it->second;
  }  // retrieve()
  
  // ----------------------------------------------------------------------
  
  void
  ParameterSet::insert( bool okay_to_replace, std::string const& name, Entry const& value ) {
    table::iterator  it = tbl.find(name);
  
    if( it == tbl.end() )  {
      if( ! tbl.insert( std::make_pair(name, value) ).second )
        throw ParameterSetError( "can't insert" + name );
    }
    else if( okay_to_replace )  {
      it->second = value;
    }
  }  // insert()
  
  // ----------------------------------------------------------------------
  // copy without overwriting
  // ----------------------------------------------------------------------
  
  void
  ParameterSet::augment( ParameterSet const& from ) {
    if( & from == this )
      return;
  
    for( table::const_iterator b = from.tbl.begin(), e = from.tbl.end(); b != e; ++b ) {
      this->insert(false, b->first, b->second);
    }
  }  // augment()
  
  // ----------------------------------------------------------------------
  // coding
  // ----------------------------------------------------------------------
  
  std::string
  ParameterSet::toString() const {
    std::string rep;
    for( table::const_iterator b = tbl.begin(), e = tbl.end(); b != e; ++b ) {
      if( b != tbl.begin() )
        rep += ';';
      rep += (b->first + '=' + b->second.toString());
    }
  
    return '<' + rep + '>';
  }  // to_string()
  
  // ----------------------------------------------------------------------
  
  std::string
  ParameterSet::toStringOfTracked() const {
    std::string  rep = "<";
    bool need_sep = false;
    for( table::const_iterator b = tbl.begin(), e = tbl.end(); b != e; ++b ) {
      if( b->second.isTracked() )  {
        if( need_sep )
          rep += ';';
        rep += (b->first + '=' + b->second.toString());
        need_sep = true;
      }
    }
  
    return rep + '>';
  }  // to_string()
  
  // ----------------------------------------------------------------------
  
  bool
  ParameterSet::fromString( std::string const& from ) {
    std::vector<std::string> temp;
    if( ! split(std::back_inserter(temp), from, '<', ';', '>') )
      return false;
  
    tbl.clear();  // precaution
    for( std::vector<std::string>::const_iterator b = temp.begin(), e = temp.end(); b != e; ++b ) {
      // locate required name/value separator
      std::string::const_iterator  q
        = std::find( b->begin(), b->end(), '=' );
      if( q == b->end() )
        return false;
  
      // form name unique to this ParameterSet
      std::string  name = std::string( b->begin(), q );
      if( tbl.find(name) != tbl.end() )
        return false;
  
      // form value and insert name/value pair
      Entry  value( std::string(q+1, b->end()) );
      if( ! tbl.insert( std::make_pair(name, value) ).second )
        return false;
    }
  
    return true;
  }  // from_string()
  
  // ----------------------------------------------------------------------
  // value accessors
  // ----------------------------------------------------------------------
  
  // ----------------------------------------------------------------------
  // Bool
  
  bool
  ParameterSet::getBool( std::string const& name ) const {
    return retrieve(name).getBool();
  }
  
  // ----------------------------------------------------------------------
  // Int32
  
  int
  ParameterSet::getInt32( std::string const& name ) const {
    return retrieve(name).getInt32();
  }
  
  // ----------------------------------------------------------------------
  // vInt32
  
  std::vector<int>
  ParameterSet::getVInt32( std::string const& name ) const {
    return retrieve(name).getVInt32();
  }
  
  // ----------------------------------------------------------------------
  // Uint32
  
  unsigned int
  ParameterSet::getUInt32( std::string const& name ) const {
    return retrieve(name).getUInt32();
  }
  
  // ----------------------------------------------------------------------
  // vUint32
  
  std::vector<unsigned int>
  ParameterSet::getVUInt32( std::string const& name ) const {
    return retrieve(name).getVUInt32();
  }
  
  // ----------------------------------------------------------------------
  // Double
  
  double
  ParameterSet::getDouble( std::string const& name ) const {
    return retrieve(name).getDouble();
  }
  
  // ----------------------------------------------------------------------
  // vDouble
  
  std::vector<double>
  ParameterSet::getVDouble( std::string const& name ) const {
    return retrieve(name).getVDouble();
  }
  
  // ----------------------------------------------------------------------
  // String
  
  std::string
  ParameterSet::getString( std::string const& name ) const {
    return retrieve(name).getString();
  }
  
  // ----------------------------------------------------------------------
  // vString
  
  std::vector<std::string>
  ParameterSet::getVString( std::string const& name ) const {
    return retrieve(name).getVString();
  }
  
  // ----------------------------------------------------------------------
  // ParameterSet
  
  ParameterSet
  ParameterSet::getPSet( std::string const& name ) const {
    return retrieve(name).getPSet();
  }
  
  // ----------------------------------------------------------------------
  // vPSet
  
  std::vector<ParameterSet>
  ParameterSet::getVPSet( std::string const& name ) const {
    return retrieve(name).getVPSet();
  }
  
} // namespace edm
// ----------------------------------------------------------------------

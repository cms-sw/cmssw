#ifndef DDL_Map_H
#define DDL_Map_H

#include "DetectorDescription/Parser/src/DDXMLElement.h"
#include "DetectorDescription/Core/interface/DDMap.h"

#include <vector>
#include <map>
#include <string>

// Boost parser, spirit, for parsing the std::vector elements.
#include "boost/spirit/include/classic.hpp"

namespace boost { namespace spirit { namespace classic { } } }

class Mapper : public boost::spirit::classic::grammar<Mapper> {
public:
  Mapper(DDLElementRegistry *registry) : registry_(registry) { };
  ~Mapper() { };
  template <typename ScannerT> struct definition;
private:
  DDLElementRegistry *registry_;
};

class MapPair {
public:
  MapPair(DDLElementRegistry *registry) : registry_(registry) { };
  ~MapPair() { };
  void operator()(char const* str, char const* end) const;
private:
  DDLElementRegistry *registry_;
};

class MapMakeName {
public:
  MapMakeName(DDLElementRegistry *registry) : registry_(registry) { };
  ~MapMakeName() { };
  void operator()(char const* str, char const* end) const;
private:
  DDLElementRegistry *registry_;
};

class MapMakeDouble {
public:
  MapMakeDouble(DDLElementRegistry *registry) : registry_(registry) { };
  ~MapMakeDouble() { };
  void operator()(char const* str, char const* end) const;
private:
  DDLElementRegistry *registry_;
};

///  DDLMap handles Map container.
/** @class DDLMap
 * @author Michael Case
 *
 *  DDLMap.h  -  description
 *  -------------------
 *  begin: Fri Nov 28, 2003
 *  email: case@ucdhep.ucdavis.edu
 *
 *
 *  This is the Map container.  It is a c++ stye std::map <std::string, double> and
 *  has a name associated with the Map for the DDD name-reference system.
 *
 */
class DDLMap : public DDXMLElement
{
  friend class MapPair;
  friend class MapMakeName;
  friend class MapMakeDouble;

public:

  DDLMap( DDLElementRegistry* myreg );

  virtual ~DDLMap( void );

  void preProcessElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv );

  void processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv );

  ReadMapType<std::map<std::string,double> > & getMapOfMaps( void );

private:
  dd_map_type pMap;
  ReadMapType<std::map<std::string,double> > pMapMap;
  double pDouble;
  std::string pName;
  std::string pNameSpace;

  void errorOut( const char* str );

  void do_pair( char const* str, char const* end );

  void do_makeName( char const* str, char const* end );

  void do_makeDouble( char const* str, char const* end );
};

#endif

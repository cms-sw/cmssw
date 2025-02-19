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
  Mapper() { };
  ~Mapper() { };
  template <typename ScannerT> struct definition;
};

class MapPair {
public:
  MapPair() { };
  ~MapPair() { };
  void operator()(char const* str, char const* end) const;
};

class MapMakeName {
public:
  MapMakeName() { };
  ~MapMakeName() { };
  void operator()(char const* str, char const* end) const;
};

class MapMakeDouble {
public:
  MapMakeDouble() { };
  ~MapMakeDouble() { };
  void operator()(char const* str, char const* end) const;
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

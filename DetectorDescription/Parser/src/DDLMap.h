#ifndef DDL_Map_H
#define DDL_Map_H

#include <map>
#include <string>
#include <vector>

#include "DetectorDescription/Core/interface/DDReadMapType.h"
#include "DetectorDescription/Core/interface/DDMap.h"
#include "DetectorDescription/Parser/src/DDXMLElement.h"

class DDCompactView;
class DDLElementRegistry;

class MapPair;
class MapMakeName;
class MapMakeDouble;

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
class DDLMap final : public DDXMLElement
{
  friend class MapPair;
  friend class MapMakeName;
  friend class MapMakeDouble;

public:

  DDLMap( DDLElementRegistry* myreg );

  void preProcessElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv ) override;
  void processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv ) override;

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

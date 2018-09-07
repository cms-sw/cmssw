#include "DetectorDescription/Parser/src/DDLMap.h"
#include "DetectorDescription/Core/interface/ClhepEvaluator.h"
#include "DetectorDescription/Parser/interface/DDLElementRegistry.h"

// Boost parser, spirit, for parsing the std::vector elements.
#include "boost/spirit/home/classic/core/non_terminal/grammar.hpp"
#include "boost/spirit/include/classic.hpp"

#include <cstddef>
#include <utility>

class DDCompactView;

class MapPair {
public:
  MapPair(DDLMap* iMap):map_{iMap} { };
  void operator()(char const* str, char const* end) const;
private:
  DDLMap* map_;
};

class MapMakeName {
public:
  MapMakeName(DDLMap* iMap):map_{iMap} { };
  void operator()(char const* str, char const* end) const;
private:
  DDLMap* map_;
};

class MapMakeDouble {
public:
  MapMakeDouble(DDLMap* iMap): map_{iMap} { };
  void operator()(char const* str, char const* end) const;
private:
  DDLMap* map_;
};

class Mapper : public boost::spirit::classic::grammar<Mapper> {
public:
  Mapper(DDLMap* iMap):map_{iMap} { };
  template <typename ScannerT> struct definition;

  MapPair mapPair() const { return MapPair(map_); }
  MapMakeName mapMakeName() const { return MapMakeName(map_); }
  MapMakeDouble mapMakeDouble() const { return MapMakeDouble(map_); }

private:
  DDLMap* map_;
};


namespace boost { namespace spirit { namespace classic {} } }

using namespace boost::spirit::classic;

DDLMap::DDLMap( DDLElementRegistry* myreg )
  : DDXMLElement( myreg )
{}

template <typename ScannerT> struct Mapper::definition
{
  definition(Mapper const& self)
    {
      mapSet
	=   ppair[self.mapPair()]
	>> *((',' >> ppair)[self.mapPair()])
	;
      
      ppair
	=   name
	>> ch_p('=') >> value
	;
      
      name
	=   (alpha_p >> *alnum_p)[self.mapMakeName()]
	;
      
      value
	=   (+(anychar_p - ','))[self.mapMakeDouble()]
	;     
    }

  rule<ScannerT> mapSet, ppair, name, value;
    
  rule<ScannerT> const&
  start() const { return mapSet; }    
};

void
MapPair::operator() (char const* str, char const* end) const
{ 
  map_->do_pair(str, end);
}

void
MapMakeName::operator() (char const* str, char const* end) const
{
  map_->do_makeName(str, end);
}

void
MapMakeDouble::operator() (char const* str, char const* end)const
{
  map_->do_makeDouble(str, end);
}

void
DDLMap::preProcessElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv )
{
  pName = "";
  pMap.clear();
  //pMapMap.clear(); only the DDLAlgorithm is allowed to clear this guy!
  pDouble = 0.0;
  pNameSpace = nmspace;
}

void
DDLMap::processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv )
{
  std::string tTextToParse = getText();
  DDXMLAttribute atts = getAttributeSet();
  std::string tName = atts.find("name")->second;

  if (tTextToParse.empty())
  {
    errorOut("No std::string to parse!");
  }

  // NOT IMPLEMENTED YET
  if (atts.find("type") != atts.end() && atts.find("type")->second == "string")
  {
    errorOut("Map of type std::string is not supported yet.");
  }

  Mapper mapGrammar{this};
  
  pMap.clear();

  parse_info<> info = boost::spirit::classic::parse(tTextToParse.c_str(), mapGrammar >> end_p, space_p);
  if (!info.full)
  {
    errorOut("Does not conform to name=value, name=value... etc. of ddl Map element.");
  }

  if (parent() == "Algorithm" || parent() == "SpecPar")
  {
    pMapMap[tName] = pMap;
  }
  else if (parent() == "ConstantsSection" || parent() == "DDDefinition") 
  {
    dd_map_type tMap;
    for (std::map<std::string, double>::const_iterator it = pMap.begin(); it != pMap.end(); ++it)
    {
      tMap[it->first] = it->second;
    }
    DDMap m ( getDDName(pNameSpace) , std::make_unique<dd_map_type>( tMap ));
    // clear the map of maps, because in these elements we only have ONE at a time.
    pMapMap.clear(); 
  }

  std::string nEntries = atts.find("nEntries")->second;
  if (pMap.size() != 
      size_t(myRegistry_->evaluator().eval(pNameSpace, nEntries)))
  {
    errorOut("Number of entries found in Map text does not match number in attribute nEntries.");
  }
  clear();
}

void
DDLMap::do_pair( char const* str, char const* end )
{
  pMap[pName] = pDouble;
}

void
DDLMap::do_makeName( char const* str, char const* end )    
{
  pName = std::string(str, end); 
}

void
DDLMap::do_makeDouble( char const* str, char const* end )
{
  std::string ts(str, end);
  pDouble = myRegistry_->evaluator().eval(pNameSpace, ts);
}

void
DDLMap::errorOut( const char* str )
{
  std::string msg("\nDDLMap: Failed to parse the following: \n");
  msg+= std::string(str);
  msg+="\n as a Map element (comma separated list of name=value).";
  throwError(msg);
}

ReadMapType< std::map<std::string,double> > &
DDLMap::getMapOfMaps( void ) 
{
  return pMapMap;
}

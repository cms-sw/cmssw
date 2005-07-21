/***************************************************************************
                          DDLMap.cc  -  description
                             -------------------
    begin                : Friday Nov. 21, 2003
    email                : case@ucdhep.ucdavis.edu
 ***************************************************************************/


namespace std{} using namespace std;
namespace boost { namespace spirit {} } using namespace boost::spirit;

// Boost parser, spirit, for parsing the vector elements.
#include "boost/spirit/core.hpp"

// Parser parts
#include "DetectorDescription/Parser/interface/DDLMap.h"
#include "DetectorDescription/Parser/interface/DDLElementRegistry.h"

// other DD parts
#include "DetectorDescription/Core/interface/DDMap.h"
#include "DetectorDescription/Base/interface/DDdebug.h"
#include "DetectorDescription/Base/interface/DDException.h"
#include "CLHEP/Units/SystemOfUnits.h"
#include "DetectorDescription/ExprAlgo/interface/ExprEvalSingleton.h"

#include <map>
#include <string>

//  The "real" DDLMap members.
DDLMap::DDLMap()
{
}

DDLMap::~DDLMap()
{
}
 
template <typename ScannerT> struct Mapper::definition
  {
    definition(Mapper const& self)
    {
      mapSet
	=   ppair[MapPair()]
	  >> *((',' >> ppair)[MapPair()])
	;
      
      ppair
	=   name
	  >> ch_p('=') >> value
	;
      
      name
	=   (alpha_p >> *alnum_p)[MapMakeName()]
	;
      
      value
	=   (+(anychar_p - ','))[MapMakeDouble()]
	;
      
    }

    rule<ScannerT> mapSet, ppair, name, value;
    
    rule<ScannerT> const&
    start() const { return mapSet; }    
};


void MapPair::operator() (char const* str, char const* end) const
{ 
  DDLMap* myDDLMap = dynamic_cast < DDLMap* > (DDLElementRegistry::instance()->getElement("Map"));
  myDDLMap->do_pair(str, end);
}

void MapMakeName::operator() (char const* str, char const* end) const
{
  DDLMap* myDDLMap = dynamic_cast < DDLMap* > (DDLElementRegistry::instance()->getElement("Map"));
  myDDLMap->do_makeName(str, end);
}

void MapMakeDouble::operator() (char const* str, char const* end)const
{
  DDLMap* myDDLMap = dynamic_cast < DDLMap* > (DDLElementRegistry::instance()->getElement("Map"));
  myDDLMap->do_makeDouble(str, end);
}

void DDLMap::preProcessElement (const string& name, const string& nmspace)
{
  pName = "";
  pMap.clear() ;
  //pMapMap.clear(); only the DDLAlgorithm is allowed to clear this guy!
  pDouble = 0.0;
  pNameSpace = nmspace;
}

void DDLMap::processElement (const string& name, const string& nmspace)
{
  DCOUT_V('P', "DDLMap::processElement started");

  string tTextToParse = getText();
  DDXMLAttribute atts = getAttributeSet();
  string tName = atts.find("name")->second;

  if (tTextToParse.size() == 0)
    {
      errorOut("No string to parse!");
    }

  // NOT IMPLEMENTED YET
  if (atts.find("type") != atts.end() && atts.find("type")->second == "string")
    {
      errorOut("Map of type string is not supported yet.");
    }

  Mapper mapGrammar;
  
  pMap.clear();

  parse_info<> info = boost::spirit::parse(tTextToParse.c_str(), mapGrammar, space_p);
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
      dd_map_type * tMap = new dd_map_type;
      for (map<string, double>::const_iterator it = pMap.begin(); it != pMap.end(); it++)
	{
	  (*tMap)[it->first] = it->second;
	}
      DDMap m ( getDDName(pNameSpace) , tMap);
      // clear the map of maps, because in these elements we only have ONE at a time.
      pMapMap.clear(); 
    }

  string nEntries = atts.find("nEntries")->second;
  if (pMap.size() != 
      size_t(ExprEvalSingleton::instance().eval(pNameSpace, nEntries)))
    {
      errorOut("Number of entries found in Map text does not match number in attribute nEntries.");
    }
  clear();
  
  DCOUT_V('P', "DDLMap::processElement completed");
}

void DDLMap::do_pair(char const* str, char const* end)
{
  pMap[pName] = pDouble;
}

void DDLMap::do_makeName(char const* str, char const* end)    
{
  pName = string(str, end); 
}

void DDLMap::do_makeDouble(char const* str, char const* end)
{
  string ts(str, end);
  try {
    pDouble = ExprEvalSingleton::instance().eval(pNameSpace, ts);
  }
  catch ( ... ) {
    string e("In makeDouble of DDLMap");
    e+= " failed to evaluate " + ts + " using ExprEvalSingleton in MakeDouble";
    errorOut(e.c_str());
  }
}

void DDLMap::errorOut(const char* str)
{
     string msg("\nDDLMap: Failed to parse the following: \n");
     msg+= string(str);
     msg+="\n as a Map element (comma separated list of name=value).";
     throwError(msg);
}

ReadMapType< std::map<std::string,double> > & DDLMap::getMapOfMaps() 
{
  return pMapMap;
}

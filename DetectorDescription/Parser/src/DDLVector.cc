#include "DetectorDescription/Parser/src/DDLVector.h"

#include <cstddef>
#include <map>
#include <memory>
#include <utility>

#include "DetectorDescription/Core/interface/DDStrVector.h"
#include "DetectorDescription/Core/interface/DDVector.h"
#include "DetectorDescription/Core/interface/ClhepEvaluator.h"
#include "DetectorDescription/Parser/interface/DDLElementRegistry.h"
#include "DetectorDescription/Parser/src/DDXMLElement.h"

class DDCompactView;

namespace {
template<typename F>
void parse(char const* str, F&& f) {

  auto ptr = str;

  while( *ptr != 0) {
    //remove any leading spaces
    while( std::isspace(*ptr) ) {
      ++ptr;
    }
    char const* strt = ptr;

    //find either the end of the char array
    // or a comma. Spaces are allowed
    // between characters.
    while( (*ptr != 0) and
           (*ptr !=',')) {++ptr;}
    char const* end = ptr;
    if(*ptr == ',') {++ptr;}

    if(strt == end) {
      break;
    }

    //strip off any ending spaces
    while(strt != end-1 and
          std::isspace(*(end-1)) ) {
      --end;
    }
    f(strt,end);
  }
}
}


bool
DDLVector::parse_numbers(char const* str)
{
  parse(str, [this](char const* st, char const* end) {
      do_makeDouble(st, end);
    });
  return true;
}

bool
DDLVector::parse_strings(char const* str)
{
  parse(str,[this](char const* st, char const* end) {
      do_makeString(st,end);
    });
  return true;
}

DDLVector::DDLVector( DDLElementRegistry* myreg )
  : DDXMLElement( myreg )
{}

void
DDLVector::preProcessElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv )
{
  pVector.clear();
  pStrVector.clear();
  pNameSpace = nmspace;
}

void
DDLVector::processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv )
{
  DDXMLAttribute atts = getAttributeSet();
  bool isNumVec((atts.find("type") == atts.end() 
		 || atts.find("type")->second == "numeric")
		?  true : false);
  bool isStringVec((!isNumVec && atts.find("type") != atts.end() 
		    && atts.find("type")->second == "string")
		   ? true : false);
  std::string tTextToParse = getText();

  if (tTextToParse.empty()) {
    errorOut(" EMPTY STRING ");
  }
  
  if (isNumVec) {
    if (!parse_numbers(tTextToParse.c_str())) {
      errorOut(tTextToParse.c_str());
    }
  }
  else if (isStringVec) {
    if (!parse_strings(tTextToParse.c_str())) {
      errorOut(tTextToParse.c_str());
    }
  }
  else {
    errorOut("Unexpected std::vector type. Only \"numeric\" and \"string\" are allowed.");
  }

  if (parent() == "Algorithm" || parent() == "SpecPar")
  {
    if (isNumVec) {
      pVecMap[atts.find("name")->second] = pVector;
    }
    else if (isStringVec) {
      pStrVecMap[atts.find("name")->second] = pStrVector;
    }
    size_t expNEntries = 0;
    if (atts.find("nEntries") != atts.end()) {
      std::string nEntries = atts.find("nEntries")->second;
      expNEntries = size_t (myRegistry_->evaluator().eval(pNameSpace, nEntries));
    }
    if ( (isNumVec && pVector.size() != expNEntries)
	 || (isStringVec && pStrVector.size() != expNEntries) )
    {
      std::string msg ("Number of entries found in Vector text does not match number in attribute nEntries.");
      msg += "\n\tnEntries = " + atts.find("nEntries")->second;
      msg += "\n------------------text---------\n";
      msg += tTextToParse;
      msg += "\n------------------text---------\n";
      errorOut(msg.c_str());
    }
  }
  else if (parent() == "ConstantsSection" || parent() == "DDDefinition")
  {
    if (atts.find("type") == atts.end() || atts.find("type")->second == "numeric") {
      DDVector v( getDDName(nmspace), std::make_unique<std::vector<double>>( pVector ));
    }
    else {
      DDStrVector v( getDDName(nmspace), std::make_unique<std::vector<std::string>>( pStrVector ));
    }
  }
  clear();
}

ReadMapType< std::vector<double> > &
DDLVector::getMapOfVectors( void ) 
{
  return pVecMap;
}

ReadMapType< std::vector<std::string> > &
DDLVector::getMapOfStrVectors( void )
{
  return pStrVecMap;
}

void
DDLVector::do_makeDouble( char const* str, char const* end )
{
  std::string ts(str, end);
  double td = myRegistry_->evaluator().eval(pNameSpace, ts);
  pVector.emplace_back(td);
}

void
DDLVector::do_makeString( char const* str, char const* end )
{
  std::string ts(str, end);
  pStrVector.emplace_back(ts);
}

void
DDLVector::errorOut( const char* str ) const
{
  std::string e("Failed to parse the following: \n");
  e+= std::string(str);
  e+="\n as a Vector element (comma separated list).";
  throwError (e);
}

void
DDLVector::clearall( void )
{
  DDXMLElement::clear();
  pVecMap.clear();
  pStrVecMap.clear();
}

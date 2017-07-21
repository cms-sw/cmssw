#include "DetectorDescription/Parser/src/DDLVector.h"

#include <stddef.h>
#include <map>
#include <memory>
#include <utility>

#include "DetectorDescription/Core/interface/DDStrVector.h"
#include "DetectorDescription/Core/interface/DDVector.h"
#include "DetectorDescription/Core/interface/ClhepEvaluator.h"
#include "DetectorDescription/Parser/interface/DDLElementRegistry.h"
#include "DetectorDescription/Parser/src/DDXMLElement.h"
#include "boost/spirit/include/classic.hpp"

class DDCompactView;

namespace boost { namespace spirit { namespace classic { } } } using namespace boost::spirit::classic;

using namespace boost::spirit;

class VectorMakeDouble
{
public:
  void operator() (char const* str, char const* end) const
    {
      ddlVector_->do_makeDouble(str, end);
    }
  
  VectorMakeDouble() {
    ddlVector_ = std::static_pointer_cast<DDLVector>(DDLGlobalRegistry::instance().getElement("Vector"));
  }
private: 
  std::shared_ptr<DDLVector> ddlVector_;
};

class VectorMakeString
{
public:
  void operator() (char const* str, char const* end) const
    {
      ddlVector_->do_makeString(str, end);
    }
  
  VectorMakeString() {
    ddlVector_ = std::static_pointer_cast<DDLVector>(DDLGlobalRegistry::instance().getElement("Vector"));
  }
private:
  std::shared_ptr<DDLVector> ddlVector_;
};

bool
DDLVector::parse_numbers(char const* str) const
{
  static VectorMakeDouble makeDouble;
  return parse(str,
	       ((+(anychar_p - ','))[makeDouble] 
		>> *(',' >> (+(anychar_p - ','))[makeDouble]))
	       >> end_p
	       , space_p).full;
}

bool
DDLVector::parse_strings(char const* str) const
{
  static VectorMakeString makeString;
  return parse(str,
	       ((+(anychar_p - ','))[makeString] 
		>> *(',' >> (+(anychar_p - ','))[makeString])) 
	       >> end_p
	       , space_p).full;
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

  if (tTextToParse.size() == 0) {
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
      DDVector v(getDDName(nmspace), new std::vector<double>(pVector));
    }
    else {
      DDStrVector v(getDDName(nmspace), new std::vector<std::string>(pStrVector));
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
  pVector.push_back(td);
}

void
DDLVector::do_makeString( char const* str, char const* end )
{
  std::string ts(str, end);
  pStrVector.push_back(ts);
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

/***************************************************************************
                          DDLVector.cc  -  description
                             -------------------
    begin                : Friday Nov. 21, 2003
    email                : case@ucdhep.ucdavis.edu
 ***************************************************************************/



// Parser parts
#include "DetectorDescription/Parser/interface/DDLVector.h"
#include "DetectorDescription/Parser/interface/DDLElementRegistry.h"

// other DD parts
#include "DetectorDescription/Core/interface/DDVector.h"
#include "DetectorDescription/Core/interface/DDStrVector.h"
#include "DetectorDescription/Base/interface/DDdebug.h"
#include "DetectorDescription/Base/interface/DDException.h"
#include "DetectorDescription/Base/interface/DDTypes.h"

#include "CLHEP/Units/SystemOfUnits.h"
#include "DetectorDescription/ExprAlgo/interface/ExprEvalSingleton.h"

#include <map>
#include <string>

// Boost parser, spirit, for parsing the std::vector elements.
#include "boost/spirit/core.hpp"

using namespace boost::spirit;

struct VectorMakeDouble
{
  void operator() (char const* str, char const* end) const
  {
    ddlVector_->do_makeDouble(str, end);
  }
  
  VectorMakeDouble() {
    ddlVector_ = dynamic_cast < DDLVector* > (DDLElementRegistry::instance()->getElement("Vector"));
  }
  
  DDLVector * ddlVector_;
};

struct VectorMakeString
{
  void operator() (char const* str, char const* end) const
  {
    ddlVector_->do_makeString(str, end);
  }
  
  VectorMakeString() {
    ddlVector_ = dynamic_cast < DDLVector* > (DDLElementRegistry::instance()->getElement("Vector"));
  }
  
  DDLVector * ddlVector_;
};

bool DDLVector::parse_numbers(char const* str) const
{
   static VectorMakeDouble makeDouble;
   return parse(str,
	       ((+(anychar_p - ','))[makeDouble] 
		>> *(',' >> (+(anychar_p - ','))[makeDouble]))
	       , space_p).full;
}

bool DDLVector::parse_strings(char const* str) const
{
   static VectorMakeString makeString;
   return parse(str,
	       ((+(anychar_p - ','))[makeString] 
		>> *(',' >> (+(anychar_p - ','))[makeString]))
	       , space_p).full;
}

DDLVector::DDLVector()
{
}

DDLVector::~DDLVector()
{
}
 
void DDLVector::preProcessElement (const std::string& name, const std::string& nmspace)
{
  pVector.clear();
  pStrVector.clear();
  pNameSpace = nmspace;
}

void DDLVector::processElement (const std::string& name, const std::string& nmspace)
{
  DCOUT_V('P', "DDLVector::processElement started");

  DDXMLAttribute atts = getAttributeSet();
  bool isNumVec((atts.find("type") == atts.end() 
		 || atts.find("type")->second == "numeric")
		?  true : false);
  bool isStringVec((!isNumVec && atts.find("type") != atts.end() 
		    && atts.find("type")->second == "string")
		   ? true : false);
  std::string tTextToParse = getText();
  //  cout << "tTextToParse is |"<< tTextToParse << "|" << endl;
  if (tTextToParse.size() == 0) {
    errorOut(" EMPTY STRING ");
  }
  
  if (isNumVec) {//(atts.find("type") == atts.end() || atts.find("type")->second == "numeric") {
    if (!parse_numbers(tTextToParse.c_str())) {
      errorOut(tTextToParse.c_str());
    }
  }
  else if (isStringVec) { //(atts.find("type")->second == "string") {
    if (!parse_strings(tTextToParse.c_str())) {
      errorOut(tTextToParse.c_str());
    }
  }
  else {
    errorOut("Unexpected std::vector type. Only \"numeric\" and \"string\" are allowed.");
  }


  if (parent() == "Algorithm" || parent() == "SpecPar")
    {
      if (isNumVec) { //(atts.find("type") != atts.end() || atts.find("type")->second == "numeric") {
	//	std::cout << "adding to pVecMap name= " << atts.find("name")->second << std::endl;
	//	for (std::vector<double>::const_iterator it = pVector.begin(); it != pVector.end(); ++it)
	//	  std::cout << *it << "\t" << std::endl;
	pVecMap[atts.find("name")->second] = pVector;
	//	std::cout << "size: " << pVecMap.size() << std::endl;
      }
      else if (isStringVec) { //(atts.find("type")->second == "string") {
	pStrVecMap[atts.find("name")->second] = pStrVector;
	//	cout << "it is a string, name is: " << atts.find("name")->second << endl;
      }
      size_t expNEntries = 0;
      if (atts.find("nEntries") != atts.end()) {
	std::string nEntries = atts.find("nEntries")->second;
	expNEntries = size_t (ExprEvalSingleton::instance().eval(pNameSpace, nEntries));
      }
      if (isNumVec && pVector.size() != expNEntries
	  || isStringVec && pStrVector.size() != expNEntries)
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
  DCOUT_V('P', "DDLVector::processElement completed");
}

ReadMapType< std::vector<double> > & DDLVector::getMapOfVectors() 
{
  return pVecMap;
}

ReadMapType< std::vector<std::string> > & DDLVector::getMapOfStrVectors()
{
  return pStrVecMap;
}

void DDLVector::do_makeDouble(char const* str, char const* end)
{
  std::string ts(str, end);
  try {
    double td = ExprEvalSingleton::instance().eval(pNameSpace, ts);
    pVector.push_back(td);
  }
  catch ( ... ) {
    std::string e("\n\tIn makeDouble of DDLVector failed to evaluate ");
    e+= ts;
    errorOut(ts.c_str());
  }
}

void DDLVector::do_makeString(char const* str, char const* end)
{
  std::string ts(str, end);
  pStrVector.push_back(ts);
}

void DDLVector::errorOut(const char* str) const
{
     std::string e("Failed to parse the following: \n");
     e+= std::string(str);
     e+="\n as a Vector element (comma separated list).";
     throwError (e);
}

void DDLVector::clearall()
{
  DDXMLElement::clear();
  pVecMap.clear();
  pStrVecMap.clear();
}

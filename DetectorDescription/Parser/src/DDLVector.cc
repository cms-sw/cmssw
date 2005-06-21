/***************************************************************************
                          DDLVector.cc  -  description
                             -------------------
    begin                : Friday Nov. 21, 2003
    email                : case@ucdhep.ucdavis.edu
 ***************************************************************************/


namespace std{} using namespace std;
namespace boost { namespace spirit {} } using namespace boost::spirit;

// Parser parts
#include "DetectorDescription/DDParser/interface/DDLVector.h"
#include "DetectorDescription/DDParser/interface/DDLElementRegistry.h"

// other DD parts
#include "DetectorDescription/DDCore/interface/DDVector.h"
#include "DetectorDescription/DDCore/interface/DDStrVector.h"
#include "DetectorDescription/DDBase/interface/DDdebug.h"
#include "DetectorDescription/DDBase/interface/DDException.h"
#include "DetectorDescription/DDBase/interface/DDTypes.h"

#include "CLHEP/Units/SystemOfUnits.h"
#include "DetectorDescription/DDExprAlgo/interface/ExprEvalSingleton.h"

#include <map>
#include <string>

// Boost parser, spirit, for parsing the vector elements.
#include "boost/spirit/core.hpp"

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
 
void DDLVector::preProcessElement (const string& name, const string& nmspace)
{
  pVector.clear();
  pStrVector.clear();
  pNameSpace = nmspace;
}

void DDLVector::processElement (const string& name, const string& nmspace)
{
  DCOUT_V('P', "DDLVector::processElement started");

  DDXMLAttribute atts = getAttributeSet();
  bool isNumVec((atts.find("type") == atts.end() 
		 || atts.find("type")->second == "numeric")
		?  true : false);
  bool isStringVec((!isNumVec && atts.find("type") != atts.end() 
		    && atts.find("type")->second == "string")
		   ? true : false);
  string tTextToParse = getText();
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
    errorOut("Unexpected vector type. Only \"numeric\" and \"string\" are allowed.");
  }


  if (parent() == "Algorithm" || parent() == "SpecPar")
    {
      if (isNumVec) { //(atts.find("type") != atts.end() || atts.find("type")->second == "numeric") {
	//	cout << "adding to pVecMap name= " << atts.find("name")->second << endl;
	//	for (vector<double>::const_iterator it = pVector.begin(); it != pVector.end(); it++)
	//	  cout << *it << "\t" << endl;
	pVecMap[atts.find("name")->second] = pVector;
	//	cout << "size: " << pVecMap.size() << endl;
      }
      else if (isStringVec) { //(atts.find("type")->second == "string") {
	pStrVecMap[atts.find("name")->second] = pStrVector;
      }
      size_t expNEntries = 0;
      if (atts.find("nEntries") != atts.end()) {
	string nEntries = atts.find("nEntries")->second;
	expNEntries = size_t (ExprEvalSingleton::instance().eval(pNameSpace, nEntries));
      }
      if (isNumVec && pVector.size() != expNEntries
	  || isStringVec && pStrVector.size() != expNEntries)
	{
	  string msg ("Number of entries found in Vector text does not match number in attribute nEntries.");
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
	DDVector v(getDDName(nmspace), new vector<double>(pVector));
      }
      else {
	DDStrVector v(getDDName(nmspace), new vector<string>(pStrVector));
      }
    }
  clear();
  DCOUT_V('P', "DDLVector::processElement completed");
}

ReadMapType< vector<double> > & DDLVector::getMapOfVectors() 
{
  return pVecMap;
}

ReadMapType< vector<string> > & DDLVector::getMapOfStrVectors()
{
  return pStrVecMap;
}

void DDLVector::do_makeDouble(char const* str, char const* end)
{
  string ts(str, end);
  try {
    double td = ExprEvalSingleton::instance().eval(pNameSpace, ts);
    pVector.push_back(td);
  }
  catch ( ... ) {
    string e("\n\tIn makeDouble of DDLVector failed to evaluate ");
    e+= ts;
    errorOut(ts.c_str());
  }
}

void DDLVector::do_makeString(char const* str, char const* end)
{
  string ts(str, end);
  pStrVector.push_back(ts);
}

void DDLVector::errorOut(const char* str) const
{
     string e("Failed to parse the following: \n");
     e+= string(str);
     e+="\n as a Vector element (comma separated list).";
     throwError (e);
}

void DDLVector::clearall()
{
  DDXMLElement::clear();
  pVecMap.clear();
  pStrVecMap.clear();
}

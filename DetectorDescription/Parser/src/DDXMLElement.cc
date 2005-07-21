/***************************************************************************
                          DDXMLElement.cc  -  description
                             -------------------
    begin                : Fri Mar 15 2002
    email                : case@ucdhep.ucdavis.edu
 ***************************************************************************/

namespace std{} using namespace std;

// -------------------------------------------------------------------------
// Includes
// -------------------------------------------------------------------------
#include "DetectorDescription/Parser/interface/DDXMLElement.h"
#include "DetectorDescription/Parser/interface/DDLSAX2FileHandler.h"

// DDCore dependencies
#include "DetectorDescription/Base/interface/DDdebug.h"
#include "DetectorDescription/Base/interface/DDException.h"

#include <string>
#include <algorithm>
#include <map>
#include <vector>
#include <iostream>

//#include <strstream>

// -------------------------------------------------------------------------
// Constructor/Destructor
// -------------------------------------------------------------------------

DDXMLElement::DDXMLElement() : attributes_(), text_(), autoClear_(false)
{ }

DDXMLElement::DDXMLElement(const bool& clearme) : attributes_(), text_(), autoClear_(clearme)
{ }

DDXMLElement::~DDXMLElement()
{ }

// -------------------------------------------------------------------------
// Implementation
// -------------------------------------------------------------------------

// For pre-processing, after attributes are loaded.  Default, do nothing!
void DDXMLElement::preProcessElement(const string& name, const string& nmspace)
{
  DCOUT_V('P', "DDXMLElement::preProcessElementBase default, do nothing) started-completed.");
}

// This loads the attributes into the attributes_ vector.
void DDXMLElement::loadAttributes (const string& elemName
				     , const vector<string> & names
				     , const vector<string> & values
				     , const string& nmspace)
{

  attributes_.resize(attributes_.size()+1);
  DDXMLAttribute & tAttributes =  attributes_.back();
  
  // adds attributes
  for (size_t i = 0; i < names.size(); i++)
    {
      //   tAttributes[ names[i] ] = values[i];
      tAttributes[ names[i] ] = values[i];
    }


  preProcessElement( elemName, nmspace );
  DCOUT_V('P', "DDXMLElement::loadAttributes completed. " << *this);
}

// clear data.
void DDXMLElement::clear()
{
  text_.clear();
  attributes_.clear();
  attributeAccumulator_.clear();
}

// Access to current attributes by name.
const std::string & DDXMLElement::getAttribute(const string& name) const
{
  static const std::string ldef;
  if (attributes_.size())
    return get(name, attributes_.size() - 1);
  return ldef;
}

const DDXMLAttribute& DDXMLElement::getAttributeSet(size_t aIndex) const 
{
  //  if (aIndex < attributes_.size())
  return attributes_[aIndex];  
}


const DDName DDXMLElement::getDDName(const string& defaultNS, const string& attname, size_t aIndex)
{
  if (aIndex < attributes_.size()
      && attributes_[aIndex].find(attname) != attributes_[aIndex].end()) { 
    string ns = defaultNS;
    const std::string & name = attributes_[aIndex].find(attname)->second;
    string rn = name;
    size_t foundColon= name.find(':');
    if (foundColon != std::string::npos) {
      ns = name.substr(0,foundColon);
      rn = name.substr(foundColon+1);

    }
    //    cout << "Name: " << rn << " Namespace: " << ns << endl;
    return DDName(rn, ns);
  }
  //  cout << "no " << attname <<  " default namespace: " << defaultNS << " at index " << aIndex << endl;
  string msg = "DDXMLElement:getDDName failed.  It was asked to make ";
  msg += "a DDName using attribute: " + attname;
  msg += " in position: " + itostr(int(aIndex)) + ".  There are ";
  msg += itostr(int(attributes_.size())) + " entries in the element.";
  throwError(msg);
  return DDName("justToCompile", "justToCompile"); // used to make sure that 
  // the method will compile with some compilers that are picky.
} 

string DDXMLElement::getNameSpace(const string& defaultNS, const string& attname
				  , size_t aIndex)
{
  cout << "DEPRECATED: PLEASE DO NOT USE getNameSpace ANYMORE!" << endl;
  string ns;
  const std::string & name = get(attname, aIndex);
  size_t foundColon= name.find(':');
  if (foundColon != std::string::npos)
    ns = name.substr(0,foundColon);
  else
    {
      ns = defaultNS;
    }
  return ns;
}

const string DDXMLElement::getName(const string& attname
			     , size_t aIndex)
{
  cout << "DEPRECATED: PLEASE DO NOT USE getName ANYMORE!!" << endl;
  string rn;
  const std::string & name = get(attname, aIndex);
  size_t foundColon= name.find(':');
  if (foundColon != std::string::npos)
    rn = name.substr(foundColon+1);
  {
    rn = name;
  }
  return rn;
}

// Returns a specific value from the aIndex set of attributes.
const std::string & DDXMLElement::get(const string& name, const size_t aIndex ) const
{
  static const string sts;
  if (aIndex < attributes_.size())
    {
      DDXMLAttribute::const_iterator it = attributes_[aIndex].find(name);
      if (attributes_[aIndex].end() == it)
        {
          DCOUT_V('P', "WARNING: DDXMLElement::get did not find the requested attribute: "  << name << endl << *this);
          return sts;
        }
      else
      	return (it->second);
    }
  string msg = "DDXMLElement:get failed.  It was asked for attribute " + name;
  msg += " in position " + itostr(int(aIndex)) + " when there are only ";
  msg += itostr(int(attributes_.size())) + " in the element storage.\n";
  throwError(msg);
  // meaningless...
  return sts;

}

// Returns a specific set of values as a vector of strings,
// given the attribute name.
vector<string> DDXMLElement::getVectorAttribute(const string& name)
{

  //  The idea here is that the attributeAccumulator_ is a cache of
  //  on-the-fly generation from the vector<DDXMLAttribute> and the 
  //  reason is simply to speed things up if it is requested more than once.
  vector<string> tv;
  AttrAccumType::const_iterator ita = attributeAccumulator_.find(name);
  if (ita != attributeAccumulator_.end())
    {
      tv = attributeAccumulator_[name];
      if (tv.size() < attributes_.size())
	{
	  appendAttributes(tv, name);
	}
      DCOUT_V('P', "DDXMLElement::getAttribute found attribute named " << name << " in a map of size " << size());
    }
  else
    {
      if (attributes_.size())
	{
	  appendAttributes(tv, name);
	}
      else
	{
      DCOUT_V('P', "DDXMLAttributeAccumulator::getAttribute was asked to provide a vector of values for an attribute named " << name << " but there was no such attribute.");
	      //      throw DDException(msg);
	}
    } 
  return tv;
}

// Default do-nothing processElementBases.
void DDXMLElement::processElement(const string& name, const string& nmspace)
{
  DCOUT_V('P', "DDXMLElement::processElementBase (default, do nothing) started-completed");
  loadText(string());
  if ( autoClear_ ) clear(); 
  
}

void DDXMLElement::loadText(const string& inText)
{
  text_.push_back(inText);
  //  cout << "just put a string using loadText. size is now: " << text_.size() << endl;
}

void DDXMLElement::appendText(const string& inText)
{
  static const std::string cr("\n");
  if (text_.size() > 0) {
    text_[text_.size() - 1] += cr;
    text_[text_.size() - 1] += inText ;
  } else
    {
      string msg = "DDXMLElement::appendText could not append to non-existent text.";
      throwError(msg);
    }
}

const string DDXMLElement::getText(size_t tindex) const
{
  if (tindex > text_.size()) {
    string msg = "DDXMLElement::getText tindex is greater than text_.size()).";
    throwError(msg);
  }
  return text_[tindex];
}

 bool DDXMLElement::gotText() const
 {
   if (text_.size() != 0)
     return true;
   return false;
 }

ostream & operator<<(ostream & os, const DDXMLElement & element)
{
  element.stream(os);
  return os;
}

void DDXMLElement::stream(ostream & os) const
{
  os << "Output of current element attributes:" << endl;
  for (vector<DDXMLAttribute>::const_iterator itv = attributes_.begin();
       itv != attributes_.end(); itv++)
    {
      for (DDXMLAttribute::const_iterator it = itv->begin(); 
	   it != itv->end(); it++)
	os << it->first <<  " = " << it->second << "\t";
      os << endl;
    }
}			 

void DDXMLElement::appendAttributes(vector<string> & tv
					      , const string& name)
{
  for (size_t i = tv.size(); i < attributes_.size(); i++)
    {
      DDXMLAttribute::const_iterator itnv = attributes_[i].find(name);
      if (itnv != attributes_[i].end())
	tv.push_back(itnv->second);
      else
	tv.push_back("");
    }  
}

// Number of elements accumulated.
size_t DDXMLElement::size() const
{
  return attributes_.size();
}

vector<DDXMLAttribute>::const_iterator DDXMLElement::begin()
{
  myIter_ = attributes_.begin();
  return attributes_.begin();
}

vector<DDXMLAttribute>::const_iterator DDXMLElement::end()
{
  myIter_ = attributes_.end();
  return attributes_.end();
}

vector<DDXMLAttribute>::const_iterator& DDXMLElement::operator++(int inc)
{
  myIter_ = myIter_ + inc;
  return myIter_;
}


const string& DDXMLElement::parent() const {
  DDLSAX2FileHandler* s2han = DDLParser::instance()->getDDLSAX2FileHandler();
  return s2han->parent();
}

// yet another :-)
string DDXMLElement::itostr(int i)
{
  if (i < 0) return string("-") + itostr(i * -1);

  if (i > 9)
    return itostr(i/10) + itostr(i % 10);
  else 
    {
      switch (i)
	{
	case 0: 
	  return string("0");
	  break;
	
	case 1: 
	  return string("1");
	  break;
	
	case 2: 
	  return string("2");
	  break;
	
	case 3: 
	  return string("3");
	  break;
	
	case 4: 
	  return string("4");
	  break;
	
	case 5: 
	  return string("5");
	  break;
	
	case 6: 
	  return string("6");
	  break;
	
	case 7: 
	  return string("7");
	  break;
	
	case 8: 
	  return string("8");
	  break;
	
	case 9: 
	  return string("9");
	  break;
	
	default:
	  return string(" ");
	}
    }
}

bool DDXMLElement::isEmpty () const
{
  return (attributes_.size() == 0 ? true : false);
}

void DDXMLElement::throwError(const string& keyMessage, DDException * e) const 
{
  //FIXME someday see if this will fly...  if (e == 0) { 
    string msg = keyMessage + "\n";
    msg += " Element " + DDLParser::instance()->getDDLSAX2FileHandler()->self() +"\n";
    msg += " File " + DDLParser::instance()->getCurrFileName() + ".\n";
    throw DDException(msg);
//    }
//    DDException* newe = new DDException(keyMessage);
//    e->add(newe);
//    throw (*e);
}

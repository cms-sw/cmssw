#ifndef DDPartSelection_h
#define DDPartSelection_h

#include <vector>
#include <string>
#include <iostream>
#include "DetectorDescription/DDCore/interface/DDLogicalPart.h"

#include <boost/spirit.hpp>


class DDLogicalPart;

enum ddselection_type { ddunknown,   //   ->    (should never appear!)
                        ddanynode,   //   ->    //*
	                ddanychild,  //   ->    /*
	                ddanylogp,   //   ->    //NameOfLogicalPart
	                ddanyposp,   //   ->    //NameOfLogicalPart[copyno]
	                ddchildlogp, //   ->    /NameOfLogicalPart
	                ddchildposp  //   ->    /NameOfLogicalPart[copyno]
	               };

//typedef DDRedirect<DDLogicalPartImpl> lpredir_type; // logical-part-redirection_type

struct DDPartSelRegExpLevel
{
  DDPartSelRegExpLevel(const string & ns, const string & nm, int cpn, ddselection_type t, bool isRegex=false)
  : ns_(ns), nm_(nm), copyno_(cpn), selectionType_(t), isRegex_(isRegex) { }
  std::string ns_, nm_;
  int copyno_;
  ddselection_type selectionType_;
  bool isRegex_;
};


struct DDPartSelectionLevel
{
  DDPartSelectionLevel(const DDLogicalPart &, int, ddselection_type);
  DDLogicalPart lp_;
  int copyno_;
  ddselection_type selectionType_;
};


class DDPartSelection : public vector<DDPartSelectionLevel>
{
public:
  DDPartSelection() : vector<DDPartSelectionLevel>() { }
};
/*
class DDPartSelection : public vector<DDPartSelectionLevel>
{
public:
  DDPartSelection() { }; // to use it in stl-containers
  DDPartSelection(const string & selectionstring);
  
  ~DDPartSelection() { }
  
};
*/


ostream & operator<<(ostream &, const DDPartSelection &);
ostream & operator<<(ostream &, const vector<DDPartSelection> &);

void DDTokenize2(const string & selectionstring, vector<DDPartSelRegExpLevel> & result);
void DDTokenize(const string & selectionstring, vector<DDPartSelRegExpLevel> & result);
ostream & operator<<(ostream &, const DDPartSelection &);

#endif

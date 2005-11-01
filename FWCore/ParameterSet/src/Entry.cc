// ----------------------------------------------------------------------
// $Id: Entry.cc,v 1.4 2005/06/23 19:57:23 wmtan Exp $
//
// definition of Entry's function members
// ----------------------------------------------------------------------


// ----------------------------------------------------------------------
// prerequisite source files and headers
// ----------------------------------------------------------------------

#include "FWCore/ParameterSet/interface/Entry.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/types.h"

#include <vector>
#include <string>
#include <algorithm>
#include <cctype>

namespace edm {
  namespace pset {

    struct TypeTrans {
      TypeTrans();
      
      typedef std::vector<std::string> CodeMap;
      CodeMap table_;
      std::map< std::string, char> type2Code_; 
    };
      
    // WARNING: The corresponding strings in Lexeme.cc are all lower case.
    // These strings do not in general match those.
    // This may or may not cause problems.
    TypeTrans::TypeTrans():table_(255) {
      table_['b'] = "vBool";
      table_['B'] = "bool";
      table_['i'] = "vint32";
      table_['I'] = "int32";
      table_['u'] = "vuint32";
      table_['U'] = "uint32";
      table_['s'] = "vstring";
      table_['S'] = "string";
      table_['d'] = "vdouble";
      table_['D'] = "double";
      table_['p'] = "vPSet";
      table_['P'] = "PSet";
      table_['T'] = "path";
      table_['F'] = "FileInPath";
      
      for(CodeMap::const_iterator itCode = table_.begin();
           itCode != table_.end();
           ++itCode) {
         type2Code_[*itCode] = (itCode - table_.begin());
      }
    }
  }

  static const edm::pset::TypeTrans sTypeTranslations;
  typedef std::map<std::string, char> Type2Code;
// ----------------------------------------------------------------------
// consistency-checker
// ----------------------------------------------------------------------

  void
  Entry::validate() const {
    // tracked
    if(tracked != '+' && tracked != '-')
      throw EntryError(std::string("invalid tracked code ") + tracked);
  
    // type and rep
    switch(type)  {
      case 'B':  {  // Bool
        bool  val;
        if(!decode(val, rep))
          throw EntryError(std::string("invalid Bool ") + rep);
        break;
      }
      case 'b':  {  // vBool
        std::vector<bool>  val;
        if(!decode(val, rep))
          throw EntryError(std::string("invalid vBool ") + rep);
        break;
      }
      case 'I':  {  // Int32
        int  val;
        if(!decode(val, rep))
          throw EntryError(std::string("invalid Int32 ") + rep);
        break;
      }
      case 'i':  {  // vInt32
        std::vector<int>  val;
        if(!decode(val, rep))
          throw EntryError(std::string("invalid vInt32 ") + rep);
        break;
      }
      case 'U':  {  // Uint32
        unsigned  val;
        if(!decode(val, rep))
          throw EntryError(std::string("invalid Uint32 ") + rep);
        break;
      }
      case 'u':  {  // vUint32
        std::vector<unsigned>  val;
        if(!decode(val, rep))
          throw EntryError(std::string("invalid vUint32 ") + rep);
        break;
      }
      case 'S':  {  // String
        std::string  val;
        if(!decode(val, rep))
          throw EntryError(std::string("invalid String ") + rep);
        break;
      }
      case 's':  {  // vString
        std::vector<std::string>  val;
        if(!decode(val, rep))
          throw EntryError(std::string("invalid vString ") + rep);
        break;
      }
      case 'F':  {  // FileInPath
	edm::FileInPath val;
        if(!decode(val, rep))
          throw EntryError(std::string("invalid FileInPath ") + rep);
        break;
      }
      case 'D':  {  // Double
        double  val;
        if(!decode(val, rep))
          throw EntryError(std::string("invalid Double ") + rep);
        break;
      }
      case 'd':  {  // vDouble
        std::vector<double>  val;
        if(!decode(val, rep))
          throw EntryError(std::string("invalid vDouble ") + rep);
        break;
      }
      case 'P':  {  // ParameterSet
        ParameterSet val;
        if(!decode(val, rep))
          throw EntryError(std::string("invalid ParameterSet ") + rep);
        break;
      }
      case 'p':  {  // vParameterSet
        std::vector<ParameterSet>  val;
        if(!decode(val, rep))
          throw EntryError(std::string("invalid vPSet ") + rep);
        break;
      }
      default:  {
        throw EntryError(std::string("invalid type code ") + type);
        break;
      }
    }  // switch(type)
  }  // Entry::validate()

// ----------------------------------------------------------------------
// constructors
// ----------------------------------------------------------------------

// ----------------------------------------------------------------------
// Bool

  Entry::Entry(bool val, bool is_tracked) : rep(), type('B'),
      tracked(is_tracked ? '+' : '-') {
    if(!encode(rep, val))
      throw EntryError("bad Bool value");
    validate();
  }

// ----------------------------------------------------------------------
// Int32

  Entry::Entry(int  val, bool is_tracked) : rep(), type('I'),
      tracked(is_tracked ? '+' : '-') {
    if(!encode(rep, val))
      throw EntryError("bad Int32 value");
    validate();
  }

// ----------------------------------------------------------------------
// vInt32

  Entry::Entry(std::vector<int> const& val, bool is_tracked) : rep(), type('i'),
       tracked(is_tracked ? '+' : '-') {
    if(!encode(rep, val))
      throw EntryError("bad vInt32 value");
    validate();
  }

// ----------------------------------------------------------------------
// Uint32

  Entry::Entry(unsigned val, bool is_tracked) : rep(), type('U'),
       tracked(is_tracked ? '+' : '-') {
    if(!encode(rep, val))
      throw EntryError("bad Uint32 value");
    validate();
  }

// ----------------------------------------------------------------------
// vUint32

 Entry::Entry(std::vector<unsigned> const& val, bool is_tracked) : rep(), type('u'),
       tracked(is_tracked ? '+' : '-') {
    if(!encode(rep, val))
      throw EntryError("bad vUint32 value");
    validate();
  }

// ----------------------------------------------------------------------
// Double

 Entry::Entry(double val, bool is_tracked) : rep(), type('D'),
       tracked(is_tracked ? '+' : '-') {
    if(!encode(rep, val))
      throw EntryError("bad Double value");
    validate();
  }

// ----------------------------------------------------------------------
// vDouble

  Entry::Entry(std::vector<double> const& val, bool is_tracked) : rep(), type('d'),
       tracked(is_tracked ? '+' : '-') {
    if(!encode(rep, val))
      throw EntryError("bad vDouble value");
    validate();
  }

// ----------------------------------------------------------------------
// String

  Entry::Entry(std::string const& val, bool is_tracked) : rep(), type('S'),
       tracked(is_tracked ? '+' : '-') {
    if(!encode(rep, val))
      throw EntryError("bad String value");
    validate();
  }

// ----------------------------------------------------------------------
// vString

  Entry::Entry(std::vector<std::string> const& val, bool is_tracked) :
       rep(), type('s'), tracked(is_tracked ? '+' : '-') {
    if(!encode(rep, val))
      throw EntryError("bad vString value");
    validate();
  }

// ----------------------------------------------------------------------
// FileInPath

  Entry::Entry(edm::FileInPath const& val, bool is_tracked) : rep(), type('F'),
       tracked(is_tracked ? '+' : '-') {
    if (!encode(rep, val))
      throw EntryError("bad FileInPath value");
    validate();
  }
							      

// ----------------------------------------------------------------------
// ParameterSet

  Entry::Entry(ParameterSet const& val, bool is_tracked) : rep(), type('P'),
       tracked(is_tracked ? '+' : '-') {
    if(!encode(rep, val))
      throw EntryError("bad ParameterSet value");
    validate();
  }

// ----------------------------------------------------------------------
// vPSet

  Entry::Entry(std::vector<ParameterSet> const& val, bool is_tracked) :
      rep(), type('p'), tracked(is_tracked ? '+' : '-') {
    if(!encode(rep, val))
      throw EntryError("bad vPSet value");
    validate();
  }

// ----------------------------------------------------------------------
// coded string

  Entry::Entry(std::string const& code) : rep(""), type('?'), tracked('?') {
    if(!fromString(code.begin(), code.end()))
      throw EntryError("bad encoded Entry string " + code);
    validate();
  }


  Entry::Entry(std::string const& type, std::string const& value, bool is_tracked)
      : rep(""), type('?'), tracked('?') {
    std::string codedString(is_tracked ?"-":"+");
   
    Type2Code::const_iterator itFound = sTypeTranslations.type2Code_.find(type);
    if(itFound == sTypeTranslations.type2Code_.end()) {
      throw EntryError("bad type name used for Entry : "+type);
    }
   
    codedString += itFound->second;
    codedString +='(';
    codedString += value;
    codedString +=')';
   
    if(!fromString(codedString.begin(), codedString.end()))
      throw EntryError("bad encoded Entry string " + codedString);
    validate();
   
  }

  Entry::Entry(std::string const& type, std::vector<std::string> const& value, bool is_tracked) :       rep("") , type('?'), tracked('?') {
    std::string codedString(is_tracked ?"-":"+");
   
    Type2Code::const_iterator itFound = sTypeTranslations.type2Code_.find(type);
    if(itFound == sTypeTranslations.type2Code_.end()) {
      throw EntryError("bad type name used for Entry : "+type);
    }
   
    codedString += itFound->second;
    codedString += '(';
    codedString += '{';
    std::vector<std::string>::const_iterator i = value.begin();
    std::vector<std::string>::const_iterator e = value.end();
    const std::string kSeparator(",");
    std::string sep("");
    for(; i!= e; ++i) {
      codedString += sep;
      codedString += *i;
      sep = kSeparator;
    }
    codedString += '}';
    codedString += ')';

    if(!fromString(codedString.begin(), codedString.end()))
      throw EntryError("bad encoded Entry string " + codedString);
    validate();
  }

// ----------------------------------------------------------------------
// coding
// ----------------------------------------------------------------------

  std::string
  Entry::toString() const {
    return std::string() + tracked + type + '(' + rep + ')';
  }

// ----------------------------------------------------------------------

  bool
  Entry::fromString(std::string::const_iterator const b, std::string::const_iterator const e) {
    if(static_cast<unsigned long>(e - b) < 4u || b[ 2] != '(' || e[-1] != ')')
      return false;

    tracked = b[0];
    type = b[1];
    rep = std::string(b+3, e-1);

    return true;
  }  // from_string()

// ----------------------------------------------------------------------
// value accessors
// ----------------------------------------------------------------------

// ----------------------------------------------------------------------
// Bool

  bool
  Entry::getBool() const {
    if(type != 'B')
      throw ValueError("value's type is not Bool");
    bool  val;
    if(!decode(val, rep))
      throw EntryError(std::string("invalid Bool ") + rep);
    return val;
  }


// ----------------------------------------------------------------------
// Int32

  int
  Entry::getInt32() const {
    if(type != 'I')
      throw ValueError("value's type is not Int32");
    int  val;
    if(!decode(val, rep))
      throw EntryError(std::string("invalid Int32 ") + rep);
    return val;
  }

// ----------------------------------------------------------------------
// vInt32

  std::vector<int>
  Entry::getVInt32() const {
    if(type != 'i')
      throw ValueError("value's type is not vInt32");
    std::vector<int>  val;
    if(!decode(val, rep))
      throw EntryError(std::string("invalid vInt32 ") + rep);
    return val;
  }

// ----------------------------------------------------------------------
// Uint32

  unsigned
  Entry::getUInt32() const {
    if(type != 'U')
      throw ValueError("value's type is not Uint32");
    unsigned  val;
    if(!decode(val, rep))
      throw EntryError(std::string("invalid Uint32 ") + rep);
    return val;
  }

// ----------------------------------------------------------------------
// vUint32

  std::vector<unsigned>
  Entry::getVUInt32() const {
    if(type != 'u')
      throw ValueError("value's type is not vUint32");
    std::vector<unsigned>  val;
    if(!decode(val, rep))
      throw EntryError(std::string("invalid vUint32 ") + rep);
    return val;
  }

// ----------------------------------------------------------------------
// Double

  double
  Entry::getDouble() const {
    if(type != 'D')
      throw ValueError("value's type is not Double");
    double  val;
    if(!decode(val, rep))
      throw EntryError(std::string("invalid Double ") + rep);
    return val;
  }

// ----------------------------------------------------------------------
// vDouble

  std::vector<double>
  Entry::getVDouble() const {
    if(type != 'u')
      throw ValueError("value's type is not vDouble");
    std::vector<double>  val;
    if(!decode(val, rep))
      throw EntryError(std::string("invalid vDouble ") + rep);
    return val;
  }

// ----------------------------------------------------------------------
// String

  std::string
  Entry::getString() const {
    if(type != 'S')
      throw ValueError("value's type is not String");
    std::string  val;
    if(!decode(val, rep))
      throw EntryError(std::string("invalid String ") + rep);
    return val;
  }

// ----------------------------------------------------------------------
// vString

  std::vector<std::string>
  Entry::getVString() const {
    if(type != 's')
      throw ValueError("value's type is not vString");
    std::vector<std::string>  val;
    if(!decode(val, rep))
      throw EntryError(std::string("invalid vString ") + rep);
    return val;
  }


// ----------------------------------------------------------------------
// FileInPath

  edm::FileInPath
  Entry::getFileInPath() const {
    if(type != 'F')
      throw ValueError("value's type is not FileInPath");
    edm::FileInPath val;
    if(!decode(val, rep))
      throw EntryError(std::string("invalid FileInPath ") + rep);
    return val;
  }

// ----------------------------------------------------------------------
// ParameterSet

  ParameterSet
  Entry::getPSet() const {
    if(type != 'P')
      throw ValueError("value's type is not ParameterSet");
    ParameterSet val;
    if(!decode(val, rep))
      throw EntryError(std::string("invalid ParameterSet ") + rep);
    return val;
  }

// ----------------------------------------------------------------------
// vPSet

  std::vector<ParameterSet>
  Entry::getVPSet() const {
    if(type != 'p')
      throw ValueError("value's type is not vPSet");
    std::vector<ParameterSet>  val;
    if(!decode(val, rep))
      throw EntryError(std::string("invalid vPSet ") + rep);
    return val;
  }

  }
// ----------------------------------------------------------------------

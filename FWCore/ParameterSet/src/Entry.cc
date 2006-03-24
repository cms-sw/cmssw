// ----------------------------------------------------------------------
// $Id: Entry.cc,v 1.8 2006/02/03 21:21:00 paterno Exp $
//
// definition of Entry's function members
// ----------------------------------------------------------------------


// ----------------------------------------------------------------------
// prerequisite source files and headers
// ----------------------------------------------------------------------

#include "FWCore/ParameterSet/interface/Entry.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/types.h"

#include <algorithm>
#include <cctype>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>


namespace edm {



  namespace 
  {

    // Helper functions for throwing exceptions

    void throwValueError(const char* expectedType)
    {
      throw edm::Exception(errors::Configuration, "ValueError")
	<< "values's type is not " << expectedType;
    }

    void throwEntryError(const char* expectedType, 
			 std::string const& badRep)
    {
      throw edm::Exception(errors::Configuration, "EntryError")
	<< "can not convert representation: "
	<< badRep
	<< "to value of type " << expectedType;
    }

    template <class T>
    void throwEncodeError(T const& /* value */, const char* type)
    {
      throw edm::Exception(errors::Configuration, "EncodingError")
	<< "can not encode the given value as type: " << type;
    }

//     template <class T>
//     void throwEncodeError(std::vector<T> const& values, const char* type)
//     {
//       std::ostringstream msg;
//       std::copy(values.begin(), 
// 		values.end(), 
// 		std::ostream_iterator<T>(msg, ","));
//       throw edm::Exception(errors::Configuration, "EncodingError")
// 	<< "can not encode the vector of values: "
// 	<< msg.str()
// 	<< " as type: " << type;      
//     }

  } // anonymous namespace




  namespace pset {

    struct TypeTrans {
      TypeTrans();
      
      typedef std::vector<std::string> CodeMap;
      CodeMap table_;
      std::map< std::string, char> type2Code_; 
    };
      
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
    assert ( tracked == '+' || tracked == '-' );
//     if(tracked != '+' && tracked != '-')
//       throw EntryError(std::string("invalid tracked code ") + tracked);
  
    // type and rep
    switch(type)  {
      case 'B':  {  // Bool
        bool  val;
        if (!decode(val, rep)) throwEntryError("bool", rep);
        break;
      }
      case 'b':  {  // vBool
        std::vector<bool>  val;
        if(!decode(val, rep)) throwEntryError("vector<bool>", rep);
        break;
      }
      case 'I':  {  // Int32
        int  val;
        if(!decode(val, rep)) throwEntryError("int", rep);
        break;
      }
      case 'i':  {  // vInt32
        std::vector<int>  val;
        if(!decode(val, rep)) throwEntryError("vector<int>", rep);
        break;
      }
      case 'U':  {  // Uint32
        unsigned  val;
        if(!decode(val, rep)) throwEntryError("unsigned int", rep);
        break;
      }
      case 'u':  {  // vUint32
        std::vector<unsigned>  val;
        if(!decode(val, rep)) throwEntryError("vector<unsigned int>", rep);
        break;
      }
      case 'S':  {  // String
        std::string  val;
        if(!decode(val, rep)) throwEntryError("string", rep);
        break;
      }
      case 's':  {  // vString
        std::vector<std::string>  val;
        if(!decode(val, rep)) throwEntryError("vector<string>", rep);
        break;
      }
      case 'F':  {  // FileInPath
	edm::FileInPath val;
        if(!decode(val, rep)) throwEntryError("FileInPath", rep);
        break;
      }
      case 'D':  {  // Double
        double  val;
        if(!decode(val, rep)) throwEntryError("double", rep);
        break;
      }
      case 'd':  {  // vDouble
        std::vector<double>  val;
        if(!decode(val, rep)) throwEntryError("vector<double>", rep);
        break;
      }
      case 'P':  {  // ParameterSet
        ParameterSet val;
        if(!decode(val, rep)) throwEntryError("ParameterSet", rep);
        break;
      }
      case 'p':  {  // vParameterSet
        std::vector<ParameterSet>  val;
        if(!decode(val, rep)) throwEntryError("vector<ParameterSet>", rep);
        break;
      }
      default:  {
	// We should never get here.
	assert ("Invalid type code" == 0);
        //throw EntryError(std::string("invalid type code ") + type);
        break;
      }
    }  // switch(type)
  }  // Entry::validate()

// ----------------------------------------------------------------------
// constructors
// ----------------------------------------------------------------------

// ----------------------------------------------------------------------
// Bool

  Entry::Entry(bool val, bool is_tracked) : 
    rep(), type('B'), tracked(is_tracked ? '+' : '-') 
  {
    if(!encode(rep, val)) throwEncodeError(val, "bool");
    validate();
  }

// ----------------------------------------------------------------------
// Int32

  Entry::Entry(int  val, bool is_tracked) : 
    rep(), type('I'), tracked(is_tracked ? '+' : '-') 
  {
    if(!encode(rep, val)) throwEncodeError(val, "int");
    validate();
  }

// ----------------------------------------------------------------------
// vInt32

  Entry::Entry(std::vector<int> const& val, bool is_tracked) : 
    rep(), type('i'), tracked(is_tracked ? '+' : '-') 
  {
    if(!encode(rep, val)) throwEncodeError(val, "vector<int>");
    validate();
  }

// ----------------------------------------------------------------------
// Uint32

  Entry::Entry(unsigned val, bool is_tracked) :
    rep(), type('U'), tracked(is_tracked ? '+' : '-')
  {
    if(!encode(rep, val)) throwEncodeError(val, "unsigned int");
    validate();
  }

// ----------------------------------------------------------------------
// vUint32

 Entry::Entry(std::vector<unsigned> const& val, bool is_tracked) :
   rep(), type('u'), tracked(is_tracked ? '+' : '-') 
 {
   if(!encode(rep, val)) throwEncodeError(val, "vector<unsigned int>");
    validate();
  }

// ----------------------------------------------------------------------
// Double

 Entry::Entry(double val, bool is_tracked) : 
   rep(), type('D'), tracked(is_tracked ? '+' : '-') 
 {
   if(!encode(rep, val)) throwEncodeError(val, "double");
    validate();
  }

// ----------------------------------------------------------------------
// vDouble

  Entry::Entry(std::vector<double> const& val, bool is_tracked) : 
    rep(), type('d'), tracked(is_tracked ? '+' : '-') 
  {
    if(!encode(rep, val)) throwEncodeError(val, "vector<double>");
    validate();
  }

// ----------------------------------------------------------------------
// String

  Entry::Entry(std::string const& val, bool is_tracked) :
    rep(), type('S'), tracked(is_tracked ? '+' : '-') 
  {
    if(!encode(rep, val)) throwEncodeError(val, "string");
    validate();
  }

// ----------------------------------------------------------------------
// vString

  Entry::Entry(std::vector<std::string> const& val, bool is_tracked) :
       rep(), type('s'), tracked(is_tracked ? '+' : '-') 
  {
    if(!encode(rep, val)) throwEncodeError(val, "vector<string>");
    validate();
  }

// ----------------------------------------------------------------------
// FileInPath

  Entry::Entry(edm::FileInPath const& val, bool is_tracked) : 
    rep(), type('F'), tracked(is_tracked ? '+' : '-') 
  {
    if (!encode(rep, val)) throwEncodeError(val, "FileInPath");
    validate();
  }
							      

// ----------------------------------------------------------------------
// ParameterSet

  Entry::Entry(ParameterSet const& val, bool is_tracked) : 
    rep(), type('P'), tracked(is_tracked ? '+' : '-') 
  {
    if(!encode(rep, val)) throwEncodeError(val, "ParameterSet");
    validate();
  }

// ----------------------------------------------------------------------
// vPSet

  Entry::Entry(std::vector<ParameterSet> const& val, bool is_tracked) :
      rep(), type('p'), tracked(is_tracked ? '+' : '-') 
  {
    if(!encode(rep, val)) throwEncodeError(val, "vector<ParameterSet>");
    validate();
  }

// ----------------------------------------------------------------------
// coded string

  Entry::Entry(std::string const& code) : 
    rep(""), type('?'), tracked('?') 
  {
    if(!fromString(code.begin(), code.end())) 
      throwEncodeError(code, "coded string");
    validate();
  }


  Entry::Entry(std::string const& type, std::string const& value, 
	       bool is_tracked) :
    rep(""), type('?'), tracked('?') 
  {
    std::string codedString(is_tracked ?"-":"+");
   
    Type2Code::const_iterator itFound = sTypeTranslations.type2Code_.find(type);
    if(itFound == sTypeTranslations.type2Code_.end()) 
      {
	throw edm::Exception(errors::Configuration)
	  << "bad type name used for Entry : " << type;
      }
   
    codedString += itFound->second;
    codedString +='(';
    codedString += value;
    codedString +=')';
   
    if(!fromString(codedString.begin(), codedString.end()))
      {
	throw edm::Exception(errors::Configuration)
	  <<  "bad encoded Entry string " <<  codedString;
      }
    validate();   
  }

  Entry::Entry(std::string const& type, 
	       std::vector<std::string> const& value, 
	       bool is_tracked) :
    rep("") , type('?'), tracked('?') 
  {
    std::string codedString(is_tracked ?"-":"+");
   
    Type2Code::const_iterator itFound = 
      sTypeTranslations.type2Code_.find(type);
    if(itFound == sTypeTranslations.type2Code_.end()) 
      {
	throw edm::Exception(errors::Configuration)
	  << "bad type name used for Entry : " << type;
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
      {
	throw edm::Exception(errors::Configuration)
	  << "bad encoded Entry string " << codedString;
      }
    validate();
  }

// ----------------------------------------------------------------------
// coding
// ----------------------------------------------------------------------

  std::string
  Entry::toString() const {
    return std::string() + tracked + type + '(' + rep + ')';
  }

  std::string
  Entry::toStringOfTracked() const {
    std::string result;
    result += tracked;
    result += type;
    result += '(';

    switch (type)
      {
      case 'P': // ParameterSet
	{
	  // Make sure we get the representation of the contained
	  // ParameterSet including *only* tracked parameters
	  ParameterSet val = getPSet();
	  result += val.toStringOfTracked();
	  break;
	}
      case 'p': // vector<ParameterSet>
	{
	  // Make sure we get the representation of each contained
	  // ParameterSet including *only* tracked parameters
	 std::vector<ParameterSet> whole = getVPSet();
	 std::vector<ParameterSet> onlytracked;
	 onlytracked.reserve(whole.size());
	 std::vector<ParameterSet>::const_iterator i = whole.begin();
	 std::vector<ParameterSet>::const_iterator e = whole.end();
	  for ( ; i != e; ++i )
	    {
	      ParameterSet tracked_part( i->toStringOfTracked() );
	      onlytracked.push_back(tracked_part);
	    }
	  std::string tracked_rep;
	  if(!encode(tracked_rep, onlytracked)) 
	    throwEncodeError(onlytracked, "vector<ParameterSet>");	  
	  result += tracked_rep;
	  break;
	}
      default: // everything else
	{
	  result += rep;
	  break;	  
	}
      }
    result += ')';
    return result;
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
    if (type != 'B') throwValueError("bool");
    bool  val;
    if (!decode(val, rep)) throwEntryError("bool", rep);
    return val;
  }


// ----------------------------------------------------------------------
// Int32

  int
  Entry::getInt32() const 
  {
    if(type != 'I') throwValueError("int");
    int  val;
    if(!decode(val, rep)) throwEntryError("int", rep);
    return val;
  }

// ----------------------------------------------------------------------
// vInt32

  std::vector<int>
  Entry::getVInt32() const 
  {
    if(type != 'i') throwValueError("vector<int>");
    std::vector<int>  val;
    if(!decode(val, rep)) throwEntryError("vector<int>", rep);
    return val;
  }

// ----------------------------------------------------------------------
// Uint32

  unsigned
  Entry::getUInt32() const 
  {
    if(type != 'U') throwValueError("unsigned int");
    unsigned  val;
    if(!decode(val, rep)) throwEntryError("unsigned int", rep);
    return val;
  }

// ----------------------------------------------------------------------
// vUint32

  std::vector<unsigned>
  Entry::getVUInt32() const 
  {
    if(type != 'u') throwValueError("vector<unsigned int>");
    std::vector<unsigned>  val;
    if(!decode(val, rep)) throwEntryError("vector<unsigned int>", rep);
    return val;
  }

// ----------------------------------------------------------------------
// Double

  double
  Entry::getDouble() const 
  {
    if(type != 'D') throwValueError("double");
    double  val;
    if(!decode(val, rep)) throwEntryError("double", rep);
    return val;
  }

// ----------------------------------------------------------------------
// vDouble

  std::vector<double>
  Entry::getVDouble() const 
  {
    if(type != 'd') throwValueError("vector<double>");
    std::vector<double>  val;
    if(!decode(val, rep)) throwEntryError("vector<double>", rep);
    return val;
  }

// ----------------------------------------------------------------------
// String

  std::string
  Entry::getString() const 
  {
    if(type != 'S') throwValueError("string");
    std::string  val;
    if(!decode(val, rep)) throwEntryError("string", rep);
    return val;
  }

// ----------------------------------------------------------------------
// vString

  std::vector<std::string>
  Entry::getVString() const 
  {
    if(type != 's') throwValueError("vector<string>");
    std::vector<std::string>  val;
    if(!decode(val, rep)) throwEntryError("vector<string>", rep);
    return val;
  }


// ----------------------------------------------------------------------
// FileInPath

  edm::FileInPath
  Entry::getFileInPath() const 
  {
    if(type != 'F') throwValueError("FileInPath");
    edm::FileInPath val;
    if(!decode(val, rep)) throwEntryError("FileInPath", rep);
    return val;
  }

// ----------------------------------------------------------------------
// ParameterSet

  ParameterSet
  Entry::getPSet() const 
  {
    if(type != 'P') throwValueError("ParameterSet");
    ParameterSet val;
    if(!decode(val, rep)) throwEntryError("ParameterSet", rep);
    return val;
  }

// ----------------------------------------------------------------------
// vPSet

  std::vector<ParameterSet>
  Entry::getVPSet() const 
  {
    if(type != 'p') throwValueError("vector<ParameterSet>");
    std::vector<ParameterSet>  val;
    if(!decode(val, rep)) throwEntryError("vector<ParameterSet>", rep);
    return val;
  }

}

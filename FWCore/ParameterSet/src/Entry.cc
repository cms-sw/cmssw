// ----------------------------------------------------------------------
// $Id: Entry.cc,v 1.33 2008/12/19 00:45:08 wmtan Exp $
//
// definition of Entry's function members
// ----------------------------------------------------------------------


// ----------------------------------------------------------------------
// prerequisite source files and headers
// ----------------------------------------------------------------------

#include "FWCore/ParameterSet/interface/Entry.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/types.h"

#include <map>
#include <sstream>
#include <ostream>
#include <assert.h>
#include <iostream>

namespace edm {
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
      table_['l'] = "vint64";
      table_['L'] = "int64";
      table_['x'] = "vuint64";
      table_['X'] = "uint64";
      table_['s'] = "vstring";
      table_['S'] = "string";
      table_['d'] = "vdouble";
      table_['D'] = "double";
      table_['p'] = "vPSet";
      table_['P'] = "PSet";
      table_['T'] = "path";
      table_['F'] = "FileInPath";
      table_['t'] = "InputTag";
      table_['v'] = "VInputTag";
      table_['e'] = "VEventID";
      table_['E'] = "EventID";
      table_['m'] = "VLuminosityBlockID";
      table_['M'] = "LuminosityBlockID";



      
      for(CodeMap::const_iterator itCode = table_.begin(), itCodeEnd = table_.end();
           itCode != itCodeEnd;
           ++itCode) {
         type2Code_[*itCode] = (itCode - table_.begin());
      }
    }
  }

  static const edm::pset::TypeTrans sTypeTranslations;
  typedef std::map<std::string, char> Type2Code;

  Entry::~Entry() {}

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
      case 'L':  {  // Int64
        int  val;
        if(!decode(val, rep)) throwEntryError("int64", rep);
        break;
      }
      case 'l':  {  // vInt64
        std::vector<int>  val;
        if(!decode(val, rep)) throwEntryError("vector<int64>", rep);
        break;
      }
      case 'X':  {  // Uint64
        unsigned  val;
        if(!decode(val, rep)) throwEntryError("unsigned int64", rep);
        break;
      }
      case 'x':  {  // vUint64
        std::vector<unsigned>  val;
        if(!decode(val, rep)) throwEntryError("vector<unsigned int64>", rep);
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
      case 't':  {  // InputTag
        edm::InputTag val;
        if(!decode(val, rep)) throwEntryError("InputTag", rep);
        break;
      }
      case 'v':  {  // VInputTag
        std::vector<edm::InputTag> val;
        if(!decode(val, rep)) throwEntryError("VInputTag", rep);
        break;
      }
      case 'E':  {  // EventID
        edm::EventID val; 
        if(!decode(val, rep)) throwEntryError("EventID", rep);
        break;
      }
      case 'e':  {  // VEventID
        std::vector<edm::EventID> val;
        if(!decode(val, rep)) throwEntryError("VEventID", rep);
        break;
      }
      case 'M':  {  // LuminosityBlockID
        edm::LuminosityBlockID val; 
        if(!decode(val, rep)) throwEntryError("LuminosityBlockID", rep);
        break;
      }
      case 'm':  {  // VLuminosityBlockID
        std::vector<edm::LuminosityBlockID> val;
        if(!decode(val, rep)) throwEntryError("VLuminosityBlockID", rep);
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

  Entry::Entry(std::string const& name, bool val, bool is_tracked) : 
    name_(name), rep(), type('B'), tracked(is_tracked ? '+' : '-') 
  {
    if(!encode(rep, val)) throwEncodeError("bool");
    validate();
  }

// ----------------------------------------------------------------------
// Int32

  Entry::Entry(std::string const& name, int  val, bool is_tracked) : 
    name_(name), rep(), type('I'), tracked(is_tracked ? '+' : '-') 
  {
    if(!encode(rep, val)) throwEncodeError("int");
    validate();
  }

// ----------------------------------------------------------------------
// vInt32

  Entry::Entry(std::string const& name, std::vector<int> const& val, bool is_tracked) : 
    name_(name), rep(), type('i'), tracked(is_tracked ? '+' : '-') 
  {
    if(!encode(rep, val)) throwEncodeError("vector<int>");
    validate();
  }

// ----------------------------------------------------------------------
// Uint32

  Entry::Entry(std::string const& name, unsigned val, bool is_tracked) :
    name_(name), rep(), type('U'), tracked(is_tracked ? '+' : '-')
  {
    if(!encode(rep, val)) throwEncodeError("unsigned int");
    validate();
  }

// ----------------------------------------------------------------------
// vUint32

 Entry::Entry(std::string const& name, std::vector<unsigned> const& val, bool is_tracked) :
   name_(name), rep(), type('u'), tracked(is_tracked ? '+' : '-') 
 {
   if(!encode(rep, val)) throwEncodeError("vector<unsigned int>");
    validate();
  }

// ----------------------------------------------------------------------
// Int64

  Entry::Entry(std::string const& name, boost::int64_t val, bool is_tracked) :
    name_(name), rep(), type('L'), tracked(is_tracked ? '+' : '-')
  {
    if(!encode(rep, val)) throwEncodeError("int64");
    validate();
  }

// ----------------------------------------------------------------------
// vInt64

  Entry::Entry(std::string const& name, std::vector<boost::int64_t> const& val, bool is_tracked) :
    name_(name), rep(), type('l'), tracked(is_tracked ? '+' : '-')
  {
    if(!encode(rep, val)) throwEncodeError("vector<int64>");
    validate();
  }

// ----------------------------------------------------------------------
// Uint64

  Entry::Entry(std::string const& name, boost::uint64_t val, bool is_tracked) :
    name_(name), rep(), type('X'), tracked(is_tracked ? '+' : '-')
  {
    if(!encode(rep, val)) throwEncodeError("unsigned int64");
    validate();
  }

// ----------------------------------------------------------------------
// vUint64

 Entry::Entry(std::string const& name, std::vector<boost::uint64_t> const& val, bool is_tracked) :
   name_(name), rep(), type('x'), tracked(is_tracked ? '+' : '-')
 {
   if(!encode(rep, val)) throwEncodeError("vector<unsigned int64>");
    validate();
  }

// ----------------------------------------------------------------------
// Double

 Entry::Entry(std::string const& name, double val, bool is_tracked) : 
   name_(name), rep(), type('D'), tracked(is_tracked ? '+' : '-') 
 {
   if(!encode(rep, val)) throwEncodeError("double");
    validate();
  }

// ----------------------------------------------------------------------
// vDouble

  Entry::Entry(std::string const& name, std::vector<double> const& val, bool is_tracked) : 
    name_(name), rep(), type('d'), tracked(is_tracked ? '+' : '-') 
  {
    if(!encode(rep, val)) throwEncodeError("vector<double>");
    validate();
  }

// ----------------------------------------------------------------------
// String

  Entry::Entry(std::string const& name, std::string const& val, bool is_tracked) :
    name_(name), rep(), type('S'), tracked(is_tracked ? '+' : '-') 
  {
    if(!encode(rep, val)) throwEncodeError("string");
    validate();
  }

// ----------------------------------------------------------------------
// vString

  Entry::Entry(std::string const& name, std::vector<std::string> const& val, bool is_tracked) :
       name_(name), rep(), type('s'), tracked(is_tracked ? '+' : '-') 
  {
    if(!encode(rep, val)) throwEncodeError("vector<string>");
    validate();
  }

// ----------------------------------------------------------------------
// FileInPath

  Entry::Entry(std::string const& name, edm::FileInPath const& val, bool is_tracked) : 
    name_(name), rep(), type('F'), tracked(is_tracked ? '+' : '-') 
  {
    if (!encode(rep, val)) throwEncodeError("FileInPath");
    validate();
  }
							      

// ----------------------------------------------------------------------
// InputTag

  Entry::Entry(std::string const& name, edm::InputTag const& val, bool is_tracked) :
    name_(name), rep(), type('t'), tracked(is_tracked ? '+' : '-')
  {
    if (!encode(rep, val)) throwEncodeError("InputTag");
    validate();
  }


// ----------------------------------------------------------------------
// VInputTag

  Entry::Entry(std::string const& name, std::vector<edm::InputTag> const& val, bool is_tracked) :
    name_(name), rep(), type('v'), tracked(is_tracked ? '+' : '-')
  {
    if (!encode(rep, val)) throwEncodeError("VInputTag");
    validate();
  }


// ----------------------------------------------------------------------
//  EventID

  Entry::Entry(std::string const& name, edm::EventID const& val, bool is_tracked) :
    name_(name), rep(), type('E'), tracked(is_tracked ? '+' : '-')
  {
    if (!encode(rep, val)) throwEncodeError("EventID");
    validate();
  }


// ----------------------------------------------------------------------
// VEventID

  Entry::Entry(std::string const& name, std::vector<edm::EventID> const& val, bool is_tracked) :
    name_(name), rep(), type('e'), tracked(is_tracked ? '+' : '-')
  {
    if (!encode(rep, val)) throwEncodeError("VEventID");
    validate();
  }



// ----------------------------------------------------------------------
//  LuminosityBlockID

  Entry::Entry(std::string const& name, edm::LuminosityBlockID const& val, bool is_tracked) :
    name_(name), rep(), type('M'), tracked(is_tracked ? '+' : '-')
  {
    if (!encode(rep, val)) throwEncodeError("LuminosityBlockID");
    validate();
  }


// ----------------------------------------------------------------------
// VLuminosityBlockID

  Entry::Entry(std::string const& name, std::vector<edm::LuminosityBlockID> const& val, bool is_tracked) :
    name_(name), rep(), type('m'), tracked(is_tracked ? '+' : '-')
  {
    if (!encode(rep, val)) throwEncodeError("VLuminosityBlockID");
    validate();
  }



// ----------------------------------------------------------------------
// ParameterSet

  Entry::Entry(std::string const& name, ParameterSet const& val, bool is_tracked) : 
    name_(name), rep(), type('P'), tracked(is_tracked ? '+' : '-') 
  {
    if(!encode(rep, val)) throwEncodeError("ParameterSet");
    validate();
  }

// ----------------------------------------------------------------------
// vPSet

  Entry::Entry(std::string const& name, std::vector<ParameterSet> const& val, bool is_tracked) :
      name_(name), rep(), type('p'), tracked(is_tracked ? '+' : '-') 
  {
    if(!encode(rep, val)) throwEncodeError("vector<ParameterSet>");
    validate();
  }

// ----------------------------------------------------------------------
// coded string

  Entry::Entry(std::string const& name, std::string const& code) : 
    name_(name), rep(), type('?'), tracked('?') 
  {
    if(!fromString(code.begin(), code.end())) 
      throwEncodeError("coded string");
    validate();
  }


  Entry::Entry(std::string const& name, std::string const& type, std::string const& value, 
	       bool is_tracked) :
    name_(name), rep(), type('?'), tracked('?') 
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

  Entry::Entry(std::string const& name, std::string const& type, 
	       std::vector<std::string> const& value, 
	       bool is_tracked) :
    name_(name), rep() , type('?'), tracked('?') 
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

  void
  Entry::toString(std::string& result) const {
    result.reserve(result.size() + sizeOfString());
    result += tracked;
    result += type;
    result += '(';
    result += rep;
    result += ')';
  }

  std::string
  Entry::toString() const {
    std::string result;
    toString(result);
    return result;
  }

// ----------------------------------------------------------------------

  bool
  Entry::fromString(std::string::const_iterator const b, std::string::const_iterator const e) {
    if(static_cast<unsigned int>(e - b) < 4u || b[ 2] != '(' || e[-1] != ')')

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
// Int32

  boost::int64_t
  Entry::getInt64() const
  {
    if(type != 'L') throwValueError("int64");
    boost::int64_t  val;
    if(!decode(val, rep)) throwEntryError("int64", rep);
    return val;
  }

// ----------------------------------------------------------------------
// vInt32

  std::vector<boost::int64_t>
  Entry::getVInt64() const
  {
    if(type != 'l') throwValueError("vector<int64>");
    std::vector<boost::int64_t>  val;
    if(!decode(val, rep)) throwEntryError("vector<int64>", rep);
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
// Uint64

  boost::uint64_t
  Entry::getUInt64() const
  {
    if(type != 'X') throwValueError("uint64");
    boost::uint64_t val;
    if(!decode(val, rep)) throwEntryError("uint64", rep);
    return val;
  }

// ----------------------------------------------------------------------
// vUint64

  std::vector<boost::uint64_t>
  Entry::getVUInt64() const
  {
    if(type != 'x') throwValueError("vector<uint64>");
    std::vector<boost::uint64_t>  val;
    if(!decode(val, rep)) throwEntryError("vector<uint64>", rep);
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
// InputTag

  edm::InputTag
  Entry::getInputTag() const
  {
    if(type != 't') throwValueError("InputTag");
    edm::InputTag val;
    if(!decode(val, rep)) throwEntryError("InputTag", rep);
    return val;
  }


// ----------------------------------------------------------------------
// VInputTag

  std::vector<edm::InputTag>
  Entry::getVInputTag() const
  {
    if(type != 'v') throwValueError("VInputTag");
    std::vector<edm::InputTag> val;
    if(!decode(val, rep)) throwEntryError("InputTag", rep);
    return val;
  }


// ----------------------------------------------------------------------
// EventID

  edm::EventID
  Entry::getEventID() const
  {
    if(type != 'E') throwValueError("EventID");
    edm::EventID val;
    if(!decode(val, rep)) throwEntryError("EventID", rep);
    return val;
  }

// ----------------------------------------------------------------------
// VEventID

  std::vector<edm::EventID>
  Entry::getVEventID() const
  {
    if(type != 'e') throwValueError("VEventID");
    std::vector<edm::EventID> val;
    if(!decode(val, rep)) throwEntryError("EventID", rep);
    return val;
  }


// ----------------------------------------------------------------------
// LuminosityBlockID

  edm::LuminosityBlockID
  Entry::getLuminosityBlockID() const
  {
    if(type != 'M') throwValueError("LuminosityBlockID");
    edm::LuminosityBlockID val;
    if(!decode(val, rep)) throwEntryError("LuminosityBlockID", rep);
    return val;
  }

// ----------------------------------------------------------------------
// VLuminosityBlockID

  std::vector<edm::LuminosityBlockID>
  Entry::getVLuminosityBlockID() const
  {
    if(type != 'm') throwValueError("VLuminosityBlockID");
    std::vector<edm::LuminosityBlockID> val;
    if(!decode(val, rep)) throwEntryError("LuminosityBlockID", rep);
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


  std::ostream&
  operator<< (std::ostream& os, const Entry & entry)
  {
    os << sTypeTranslations.table_[entry.typeCode()] << " " 
       << (entry.isTracked() ? "tracked " : "untracked ") <<" = "; 

    // now handle the difficult cases
    switch(entry.typeCode())
    {
      case 'P': // ParameterSet
      {
        os << entry.getPSet();
        break;
      }
      case 'p': // vector<ParameterSet>
      {
        // Make sure we get the representation of each contained
        // ParameterSet including *only* tracked parameters
        std::vector<ParameterSet> whole = entry.getVPSet();
        std::vector<ParameterSet>::const_iterator i = whole.begin();
        std::vector<ParameterSet>::const_iterator e = whole.end();
        std::string start ="";
        const std::string between(",\n");
        os << "{"<<std::endl;
         for ( ; i != e; ++i )
         {
           os <<  start<< *i;
           start = between;
         }
         if (whole.size()) {
           os << std::endl;
         }
         os << "}";
         break;
      } 
      case 'S':
      {
        os << "'" << entry.getString() << "'";
        break;
      }
      case 's':
      {
        
        os << "{";
        std::string start ="'";
        const std::string between(",'");
        std::vector<std::string> strings = entry.getVString();
        for(std::vector<std::string>::const_iterator it = strings.begin(), itEnd = strings.end();
            it != itEnd;
            ++it) {
          os << start << *it << "'";
          start = between;
        }
        os << "}";
        break;
      }
      case 'I':
      {
        os << entry.getInt32();
        break;
      }
      case 'U':
      {
        os << entry.getUInt32();
        break;
      }
      case 'v':
      {
        //VInputTag needs to be treated seperately because it is encode like
        // vector<string> rather than using the individual encodings of each InputTag
        os << "{";
        std::string start = "";
        const std::string between(",");
        std::vector<InputTag> tags = entry.getVInputTag();
        for(std::vector<InputTag>::const_iterator it = tags.begin(), itEnd = tags.end();
            it != itEnd;
            ++it) {
          os << start << it->encode();
          start = between;
        }
        os << "}";
        break;
      }
      default:
      {
        os << entry.rep;
        break;
      }
    }

    return os;
  }

    // Helper functions for throwing exceptions

    void Entry::throwValueError(const char* expectedType) const
    {
      throw edm::Exception(errors::Configuration, "ValueError")
        << "type of " << name_ << " is expected to be " << expectedType
        << " but declared as " << sTypeTranslations.table_[type];
    }

    void Entry::throwEntryError(const char* expectedType,
                         std::string const& badRep) const
    {
      throw edm::Exception(errors::Configuration, "EntryError")
        << "can not convert representation of " << name_ << ": "
        << badRep
        << " to value of type " << expectedType << " ";
    }

    void Entry::throwEncodeError(const char* type) const
    {
      throw edm::Exception(errors::Configuration, "EncodingError")
        << "can not encode " << name_ << " as type: " << type;
    }


}

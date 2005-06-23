// ----------------------------------------------------------------------
// $Id: Entry.h,v 1.2 2005/06/10 03:55:27 wmtan Exp $
//
// interface to edm::Entry and related types
// ----------------------------------------------------------------------

// ----------------------------------------------------------------------
// prolog

#ifndef  ENTRY_H
#define  ENTRY_H

// ----------------------------------------------------------------------
// prerequisite source files and headers

#include <string>
#include <stdexcept>
#include <vector>
#include <map>

// ----------------------------------------------------------------------
// contents

namespace edm {
  // forward declarations:
  class ParameterSet;

  // ----------------------------------------------------------------------
  // EntryError
  
  class EntryError : public std::runtime_error {
  public:
    explicit EntryError(std::string const& mesg)
      : std::runtime_error(mesg) {}
  
    virtual ~EntryError() throw() {}
  
  };  // EntryError
  
  // ----------------------------------------------------------------------
  // ValueError
  
  class ValueError : public std::runtime_error {
  public:
    explicit ValueError(std::string const& mesg)
      : std::runtime_error(mesg) {}
  
    virtual ~ValueError() throw() {}
  
  };  // ValueError
  
  // ----------------------------------------------------------------------
  // Entry
  
  class Entry {
  public:
    // default
    Entry() : rep(), type('?'), tracked('?') {}
  
    // Bool
    Entry(bool val, bool is_tracked);
    bool  getBool() const;
  
    // Int32
    Entry(int val, bool is_tracked);
    int  getInt32() const;
  
    // vInt32
    Entry(std::vector<int> const& val, bool is_tracked);
    std::vector<int>  getVInt32() const;
  
    // Uint32
    Entry(unsigned val, bool is_tracked);
    unsigned  getUInt32() const;
  
    // vUint32
    Entry(std::vector<unsigned> const& val, bool is_tracked);
    std::vector<unsigned>  getVUInt32() const;
  
    // Double
    Entry(double val, bool is_tracked);
    double getDouble() const;
  
    // vDouble
    Entry(std::vector<double> const& val, bool is_tracked);
    std::vector<double> getVDouble() const;
  
    // String
    Entry(std::string const& val, bool is_tracked);
    std::string getString() const;
  
    // vString
    Entry(std::vector<std::string> const& val, bool is_tracked);
    std::vector<std::string>  getVString() const;
  
    // ParameterSet
    Entry(ParameterSet const& val, bool is_tracked);
    ParameterSet getPSet() const;
  
    // vPSet
    Entry(std::vector<ParameterSet> const& val, bool is_tracked);
  
    std::vector<ParameterSet>  getVPSet() const;
  
    // coded string
    Entry(std::string const&);
    Entry(std::string const& type, std::string const& value, bool is_tracked);
    Entry(std::string const& type, std::vector<std::string> const& value, bool is_tracked);
    
    // encode
    std::string  toString() const;
  
    // access
    bool isTracked() const { return tracked == '+'; }
  
    char typeCode() const { return type; }
  
  private:
    std::string  rep;
    char         type;
    char         tracked;
  
    // verify class invariant
    void validate() const;
  
    // decode
    bool fromString(std::string::const_iterator b, std::string::const_iterator e);
  };  // Entry
  
  inline bool
  operator==(Entry const& a, Entry const& b) {
    return a.toString() == b.toString();
  }
  
  inline bool
  operator!=(Entry const& a, Entry const& b) {
    return !(a == b);
  }
} // namespace edm
  // ----------------------------------------------------------------------
  // epilog
  
#endif  // ENTRY_H

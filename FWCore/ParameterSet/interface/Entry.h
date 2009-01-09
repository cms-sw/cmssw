#ifndef FWCore_ParameterSet_Entry_h
#define FWCore_ParameterSet_Entry_h

// ----------------------------------------------------------------------
// interface to edm::Entry and related types
//
//
// The functions here are expected to go away.  The exception
// processing is not ideal and is not a good model to follow.
//
// ----------------------------------------------------------------------


#include <string>
#include <vector>
#include <iosfwd>

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Utilities/interface/InputTag.h"
//@@ not needed, but there might be trouble if we take it out
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include <boost/cstdint.hpp>

// ----------------------------------------------------------------------
// contents

namespace edm {
  // forward declarations:
  class ParameterSet;

  // ----------------------------------------------------------------------
  // Entry
  
  class Entry 
  {
  public:
    // Bool
    Entry(std::string const& name, bool val, bool is_tracked);
    bool  getBool() const;
  
    // Int32
    Entry(std::string const& name, int val, bool is_tracked);
    int  getInt32() const;
  
    // vInt32
    Entry(std::string const& name, std::vector<int> const& val, bool is_tracked);
    std::vector<int>  getVInt32() const;
  
    // Uint32
    Entry(std::string const& name, unsigned val, bool is_tracked);
    unsigned  getUInt32() const;
  
    // vUint32
    Entry(std::string const& name, std::vector<unsigned> const& val, bool is_tracked);
    std::vector<unsigned>  getVUInt32() const;
  
    // Int64
    Entry(std::string const& name, boost::int64_t val, bool is_tracked);
    boost::int64_t  getInt64() const;

    // vInt64
    Entry(std::string const& name, std::vector<boost::int64_t> const& val, bool is_tracked);
    std::vector<boost::int64_t>  getVInt64() const;

    // Uint64
    Entry(std::string const& name, boost::uint64_t val, bool is_tracked);
    boost::uint64_t  getUInt64() const;

    // vUint64
    Entry(std::string const& name, std::vector<boost::uint64_t> const& val, bool is_tracked);
    std::vector<boost::uint64_t>  getVUInt64() const;

    // Double
    Entry(std::string const& name, double val, bool is_tracked);
    double getDouble() const;
  
    // vDouble
    Entry(std::string const& name, std::vector<double> const& val, bool is_tracked);
    std::vector<double> getVDouble() const;
  
    // String
    Entry(std::string const& name, std::string const& val, bool is_tracked);
    std::string getString() const;
  
    // vString
    Entry(std::string const& name, std::vector<std::string> const& val, bool is_tracked);
    std::vector<std::string>  getVString() const;

    // FileInPath
    Entry(std::string const& name, edm::FileInPath const& val, bool is_tracked);
    edm::FileInPath getFileInPath() const;
  
    // InputTag
    Entry(std::string const& name, edm::InputTag const & tag, bool is_tracked);
    edm::InputTag getInputTag() const;

    // InputTag
    Entry(std::string const& name, std::vector<edm::InputTag> const & vtag, bool is_tracked);
    std::vector<edm::InputTag> getVInputTag() const;

    // EventID
    Entry(std::string const& name, edm::EventID const & tag, bool is_tracked);
    edm::EventID getEventID() const;

    // VEventID
    Entry(std::string const& name, std::vector<edm::EventID> const & vtag, bool is_tracked);
    std::vector<edm::EventID> getVEventID() const;

    // LuminosityBlockID
    Entry(std::string const& name, edm::LuminosityBlockID const & tag, bool is_tracked);
    edm::LuminosityBlockID getLuminosityBlockID() const;

    // VLuminosityBlockID
    Entry(std::string const& name, std::vector<edm::LuminosityBlockID> const & vtag, bool is_tracked);
    std::vector<edm::LuminosityBlockID> getVLuminosityBlockID() const;

    // ParameterSet
    Entry(std::string const& name, ParameterSet const& val, bool is_tracked);
    ParameterSet getPSet() const;
  
    // vPSet
    Entry(std::string const& name, std::vector<ParameterSet> const& val, bool is_tracked);
  
    std::vector<ParameterSet>  getVPSet() const;
  
    // coded string
    Entry(std::string const& name, std::string const&);
    Entry(std::string const& name, std::string const& type, 
          std::string const& value, bool is_tracked);
    Entry(std::string const& name, std::string const& type, 
          std::vector<std::string> const& value, bool is_tracked);
    
    ~Entry();
    // encode
    std::string  toString() const;
    std::string  toStringOfTracked() const;
    size_t sizeOfString() const {return rep.size() + 4;}
    size_t sizeOfStringOfTracked() const;
  
    // access
    bool isTracked() const { return tracked == '+'; }

    char typeCode() const { return type; }

    friend std::ostream& operator<<(std::ostream& ost, const Entry & entry);

  private:
    std::string name_;
    std::string  rep;
    mutable std::string  tracked_rep;
    char         type;
    char         tracked;
  
    // verify class invariant
    void validate() const;
  
    // decode
    bool fromString(std::string::const_iterator b, std::string::const_iterator e);

    // helpers to throw exceptions
    void throwValueError(const char* expectedType) const;
    void throwEntryError(const char* expectedType,std::string const& badRep) const;
    void throwEncodeError(const char* type) const;

  };  // Entry


  // It is not clear whether operator== should use toString() or
  // toStringOfTracked(). It only makes a differences for Entries that
  // carry ParameterSets (or vectors thereof).
  //
  // However, it seems that operator== for Entry is *nowhere used*!.
  // Thus, the code is new removed.
  //   inline bool
  //   operator==(Entry const& a, Entry const& b) {
  //     return a.toString() == b.toString();
  //   }
  
  //   inline bool
  //   operator!=(Entry const& a, Entry const& b) {
  //     return !(a == b);
  //   }
} // namespace edm

  
#endif

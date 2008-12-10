#ifndef FWCore_ParameterSet_ParameterSet_h
#define FWCore_ParameterSet_ParameterSet_h

// ----------------------------------------------------------------------
// $Id: ParameterSet.h,v 1.49 2008/12/09 16:35:57 wmtan Exp $
//
// Declaration for ParameterSet(parameter set) and related types
// ----------------------------------------------------------------------


// ----------------------------------------------------------------------
// prolog

// ----------------------------------------------------------------------
// prerequisite source files and headers

#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "FWCore/ParameterSet/interface/Entry.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include <string>
#include <map>
#include <vector>
#include <iosfwd>


// ----------------------------------------------------------------------
// contents

namespace edm {

  class ParameterSet {
  public:
    // default-construct
    ParameterSet();

    // construct from coded string
    explicit ParameterSet(std::string const&);

    ~ParameterSet();

    // identification
    ParameterSetID id() const;
    void setID(const ParameterSetID & id);

    // Entry-handling
    Entry const& retrieve(std::string const&) const;
    Entry const& retrieve(char const*) const;

    Entry const* const retrieveUntracked(std::string const&) const;
    Entry const* const retrieveUntracked(char const*) const;

    Entry const* const retrieveUnknown(std::string const&) const;
    Entry const* const retrieveUnknown(char const*) const;

    void insert(bool ok_to_replace, std::string const& , Entry const&);
    void insert(bool ok_to_replace, char const* , Entry const&);

    void augment(ParameterSet const& from); 
    // encode
    std::string toString() const;
    std::string toStringOfTracked() const;
    // decode
    bool fromString(std::string const&);


    template <typename T>
    T
    getParameter(std::string const&) const;

    template <typename T>
    T
    getParameter(char const*) const;

    template <typename T> 
    void 
    addParameter(std::string const& name, T value)
    {
      invalidate();
      insert(true, name, Entry(name, value, true));
    }

    template <typename T> 
    void 
    addParameter(char const* name, T value)
    {
      invalidate();
      insert(true, name, Entry(name, value, true));
    }

    template <typename T>
    T
    getUntrackedParameter(std::string const&, T const&) const;

    template <typename T>
    T
    getUntrackedParameter(char const*, T const&) const;

    template <typename T>
    T
    getUntrackedParameter(std::string const&) const;

    template <typename T>
    T
    getUntrackedParameter(char const*) const;

    /// The returned value is the number of new FileInPath objects
    /// pushed into the vector.
    /// N.B.: The vector 'output' is *not* cleared; new entries are
    /// added with push_back.
    std::vector<FileInPath>::size_type
    getAllFileInPaths(std::vector<FileInPath>& output) const;

    std::vector<std::string> getParameterNames() const;

    /// checks if a parameter exists
    bool exists(std::string const& parameterName) const;

    /// checks if a parameter exists as a given type
    template <typename T>
    bool existsAs(std::string const& parameterName, bool trackiness=true) const
    {
       std::vector<std::string> names = getParameterNamesForType<T>(trackiness);
       return std::find(names.begin(), names.end(), parameterName) != names.end();
    }

    void depricatedInputTagWarning(std::string const& name, std::string const& label) const;

    template <typename T>
    std::vector<std::string> getParameterNamesForType(bool trackiness = 
						      true) const
    {
      std::vector<std::string> result;
      // This is icky, but I don't know of another way in the current
      // code to get at the character code that denotes type T.
      T value = T();
      Entry type_translator("", value, trackiness);
      char type_code = type_translator.typeCode();
      
      (void)getNamesByCode_(type_code, trackiness, result);
      return result;
    }
    
    template <typename T>
    void
    addUntrackedParameter(std::string const& name, T value)
    {
      // No need to invalidate: this is modifying an untracked parameter!
      insert(true, name, Entry(name, value, false));
    }

    template <typename T>
    void
    addUntrackedParameter(char const* name, T value)
    {
      // No need to invalidate: this is modifying an untracked parameter!
      insert(true, name, Entry(name, value, false));
    }

    bool empty() const
    {
      return tbl_.empty();
    }

    ParameterSet trackedPart() const;

    // Return the names of all parameters of type ParameterSet,
    // pushing the names into the argument 'output'. Return the number
    // of names pushed into the vector. If 'trackiness' is true, we
    // return tracked parameters; if 'trackiness' is false, w return
    // untracked parameters.
    size_t getParameterSetNames(std::vector<std::string>& output,
				bool trackiness = true) const;

    // Return the names of all parameters of type
    // vector<ParameterSet>, pushing the names into the argument
    // 'output'. Return the number of names pushed into the vector. If
    // 'trackiness' is true, we return tracked parameters; if
    // 'trackiness' is false, w return untracked parameters.
    size_t getParameterSetVectorNames(std::vector<std::string>& output,
				      bool trackiness=true) const;

    // need a simple interface for python
    std::string dump() const;

    friend std::ostream & operator<<(std::ostream & os, ParameterSet const& pset);

    /// needs to be called before saving or serializing
    void fillID() const;

private:
    typedef std::map<std::string, Entry> table;
    table tbl_;

    // If the id_ is invalid, that means a new value should be
    // calculated before the value is returned. Upon construction, the
    // id_ is made valid. Updating any parameter invalidates the id_.
    mutable ParameterSetID id_;

    // make the id valid, matching the current tracked contents of
    // this ParameterSet.  This function is logically const, because
    // it affects only the cached value of the id_.
    void validate() const;

    // make the id invalid.  This function is logically const, because
    // it affects only the cached value of the id_.
    void invalidate() const;

    // get the untracked Entry object, throwing an exception if it is
    // not found.
    Entry const* getEntryPointerOrThrow_(std::string const& name) const;
    Entry const* getEntryPointerOrThrow_(char const* name) const;

    // Return the names of all the entries with the given typecode and
    // given status (trackiness)
    size_t getNamesByCode_(char code,
			   bool trackiness,
			   std::vector<std::string>& output) const;


  };  // ParameterSet

  inline
  bool
  operator==(ParameterSet const& a, ParameterSet const& b) {
    // Maybe can replace this with comparison of id_ values.
    return a.toStringOfTracked() == b.toStringOfTracked();
  }

  inline 
  bool
  operator!=(ParameterSet const& a, ParameterSet const& b) {
    return !(a == b);
  }

  // specializations
  // ----------------------------------------------------------------------
  // Bool, vBool
  
  template<>
  inline 
  bool
  ParameterSet::getParameter<bool>(std::string const& name) const {
    return retrieve(name).getBool();
  }

  // ----------------------------------------------------------------------
  // Int32, vInt32
  
  template<>
  inline 
  int
  ParameterSet::getParameter<int>(std::string const& name) const {
    return retrieve(name).getInt32();
  }

  template<>
  inline 
  std::vector<int>
  ParameterSet::getParameter<std::vector<int> >(std::string const& name) const {
    return retrieve(name).getVInt32();
  }
  
 // ----------------------------------------------------------------------
  // Int64, vInt64

  template<>
  inline
  boost::int64_t
  ParameterSet::getParameter<boost::int64_t>(std::string const& name) const {
    return retrieve(name).getInt64();
  }

  template<>
  inline
  std::vector<boost::int64_t>
  ParameterSet::getParameter<std::vector<boost::int64_t> >(std::string const& name) const {
    return retrieve(name).getVInt64();
  }

  // ----------------------------------------------------------------------
  // Uint32, vUint32
  
  template<>
  inline 
  unsigned int
  ParameterSet::getParameter<unsigned int>(std::string const& name) const {
    return retrieve(name).getUInt32();
  }
  
  template<>
  inline 
  std::vector<unsigned int>
  ParameterSet::getParameter<std::vector<unsigned int> >(std::string const& name) const {
    return retrieve(name).getVUInt32();
  }
  
  // ----------------------------------------------------------------------
  // Uint64, vUint64

  template<>
  inline
  boost::uint64_t
  ParameterSet::getParameter<boost::uint64_t>(std::string const& name) const {
    return retrieve(name).getUInt64();
  }

  template<>
  inline
  std::vector<boost::uint64_t>
  ParameterSet::getParameter<std::vector<boost::uint64_t> >(std::string const& name) const {
    return retrieve(name).getVUInt64();
  }

  // ----------------------------------------------------------------------
  // Double, vDouble
  
  template<>
  inline 
  double
  ParameterSet::getParameter<double>(std::string const& name) const {
    return retrieve(name).getDouble();
  }
  
  template<>
  inline 
  std::vector<double>
  ParameterSet::getParameter<std::vector<double> >(std::string const& name) const {
    return retrieve(name).getVDouble();
  }
  
  // ----------------------------------------------------------------------
  // String, vString
  
  template<>
  inline 
  std::string
  ParameterSet::getParameter<std::string>(std::string const& name) const {
    return retrieve(name).getString();
  }
  
  template<>
  inline 
  std::vector<std::string>
  ParameterSet::getParameter<std::vector<std::string> >(std::string const& name) const {
    return retrieve(name).getVString();
  }

  // ----------------------------------------------------------------------
  // FileInPath

  template <>
  inline
  FileInPath
  ParameterSet::getParameter<FileInPath>(std::string const& name) const {
    return retrieve(name).getFileInPath();
  }
  
  // ----------------------------------------------------------------------
  // InputTag

  template <>
  inline
  InputTag
  ParameterSet::getParameter<InputTag>(std::string const& name) const {
    Entry const& e_input = retrieve(name);
    switch (e_input.typeCode()) 
    {
      case 't':   // InputTag
        return e_input.getInputTag();
      case 'S':   // string
        std::string const& label = e_input.getString();
	depricatedInputTagWarning(name, label);
        return InputTag( label );
    }
    throw edm::Exception(errors::Configuration, "ValueError") << "type of " 
       << name << " is expected to be InputTag or string (deprecated)";

  }

  // ----------------------------------------------------------------------
  // VInputTag

  template <>
  inline
  std::vector<InputTag>
  ParameterSet::getParameter<std::vector<InputTag> >(std::string const& name) const {
    return retrieve(name).getVInputTag();
  }

  // ----------------------------------------------------------------------
  // EventID

  template <>
  inline
  EventID
  ParameterSet::getParameter<EventID>(std::string const& name) const {
    return retrieve(name).getEventID();
  }

  // ----------------------------------------------------------------------
  // VEventID

  template <>
  inline
  std::vector<EventID>
  ParameterSet::getParameter<std::vector<EventID> >(std::string const& name) const {
    return retrieve(name).getVEventID();
  }

  // ----------------------------------------------------------------------
  // LuminosityBlockID

  template <>
  inline
  LuminosityBlockID
  ParameterSet::getParameter<LuminosityBlockID>(std::string const& name) const {
    return retrieve(name).getLuminosityBlockID();
  }

  // ----------------------------------------------------------------------
  // VLuminosityBlockID

  template <>
  inline
  std::vector<LuminosityBlockID>
  ParameterSet::getParameter<std::vector<LuminosityBlockID> >(std::string const& name) const {
    return retrieve(name).getVLuminosityBlockID();
  }

  // ----------------------------------------------------------------------
  // PSet, vPSet
  
  template<>
  inline 
  ParameterSet
  ParameterSet::getParameter<ParameterSet>(std::string const& name) const {
    return retrieve(name).getPSet();
  }
  
  template<>
  inline 
  std::vector<ParameterSet>
  ParameterSet::getParameter<std::vector<ParameterSet> >(std::string const& name) const {
    return retrieve(name).getVPSet();
  }

  // untracked parameters
  
  // ----------------------------------------------------------------------
  // Bool, vBool
  
  template<>
  inline 
  bool
  ParameterSet::getUntrackedParameter<bool>(std::string const& name, bool const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getBool();
  }

  template<>
  inline
  bool
  ParameterSet::getUntrackedParameter<bool>(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getBool();
  }
  
  
  // ----------------------------------------------------------------------
  // Int32, vInt32
  
  template<>
  inline 
  int
  ParameterSet::getUntrackedParameter<int>(std::string const& name, int const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getInt32();
  }

  template<>
  inline
  int
  ParameterSet::getUntrackedParameter<int>(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getInt32();
  }

  template<>
  inline 
  std::vector<int>
  ParameterSet::getUntrackedParameter<std::vector<int> >(std::string const& name, std::vector<int> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getVInt32();
  }

  template<>
  inline
  std::vector<int>
  ParameterSet::getUntrackedParameter<std::vector<int> >(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getVInt32();
  }
  
  // ----------------------------------------------------------------------
  // Uint32, vUint32
  
  template<>
  inline 
  unsigned int
  ParameterSet::getUntrackedParameter<unsigned int>(std::string const& name, unsigned int const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getUInt32();
  }

  template<>
  inline
  unsigned int
  ParameterSet::getUntrackedParameter<unsigned int>(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getUInt32();
  }
  
  template<>
  inline 
  std::vector<unsigned int>
  ParameterSet::getUntrackedParameter<std::vector<unsigned int> >(std::string const& name, std::vector<unsigned int> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getVUInt32();
  }

  template<>
  inline
  std::vector<unsigned int>
  ParameterSet::getUntrackedParameter<std::vector<unsigned int> >(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getVUInt32();
  }


  // ----------------------------------------------------------------------
  // Uint64, vUint64

  template<>
  inline
  boost::uint64_t
  ParameterSet::getUntrackedParameter<boost::uint64_t>(std::string const& name, boost::uint64_t const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getUInt64();
  }

  template<>
  inline
  boost::uint64_t
  ParameterSet::getUntrackedParameter<boost::uint64_t>(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getUInt64();
  }

  template<>
  inline
  std::vector<boost::uint64_t>
  ParameterSet::getUntrackedParameter<std::vector<boost::uint64_t> >(std::string const& name, std::vector<boost::uint64_t> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getVUInt64();
  }

  template<>
  inline
  std::vector<boost::uint64_t>
  ParameterSet::getUntrackedParameter<std::vector<boost::uint64_t> >(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getVUInt64();
  }


  // ----------------------------------------------------------------------
  // Int64, Vint64

  template<>
  inline
  boost::int64_t
  ParameterSet::getUntrackedParameter<boost::int64_t>(std::string const& name, boost::int64_t const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getInt64();
  }

  template<>
  inline
  boost::int64_t
  ParameterSet::getUntrackedParameter<boost::int64_t>(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getInt64();
  }

  template<>
  inline
  std::vector<boost::int64_t>
  ParameterSet::getUntrackedParameter<std::vector<boost::int64_t> >(std::string const& name, std::vector<boost::int64_t> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getVInt64();
  }

  template<>
  inline
  std::vector<boost::int64_t>
  ParameterSet::getUntrackedParameter<std::vector<boost::int64_t> >(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getVInt64();
  }

  
  // ----------------------------------------------------------------------
  // Double, vDouble
  
  template<>
  inline 
  double
  ParameterSet::getUntrackedParameter<double>(std::string const& name, double const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getDouble();
  }


  template<>
  inline
  double
  ParameterSet::getUntrackedParameter<double>(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getDouble();
  }  
  
  template<>
  inline 
  std::vector<double>
  ParameterSet::getUntrackedParameter<std::vector<double> >(std::string const& name, std::vector<double> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name); return entryPtr == 0 ? defaultValue : entryPtr->getVDouble(); 
  }

  template<>
  inline
  std::vector<double>
  ParameterSet::getUntrackedParameter<std::vector<double> >(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getVDouble();
  }
  
  // ----------------------------------------------------------------------
  // String, vString
  
  template<>
  inline 
  std::string
  ParameterSet::getUntrackedParameter<std::string>(std::string const& name, std::string const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getString();
  }

  template<>
  inline
  std::string
  ParameterSet::getUntrackedParameter<std::string>(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getString();
  }
  
  template<>
  inline 
  std::vector<std::string>
  ParameterSet::getUntrackedParameter<std::vector<std::string> >(std::string const& name, std::vector<std::string> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getVString();
  }


  template<>
  inline
  std::vector<std::string>
  ParameterSet::getUntrackedParameter<std::vector<std::string> >(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getVString();
  }

  // ----------------------------------------------------------------------
  //  FileInPath

  template<>
  inline
  FileInPath
  ParameterSet::getUntrackedParameter<FileInPath>(std::string const& name, FileInPath const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getFileInPath();
  }

  template<>
  inline
  FileInPath
  ParameterSet::getUntrackedParameter<FileInPath>(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getFileInPath();
  }

  // ----------------------------------------------------------------------
  // InputTag, VInputTag

  template<>
  inline
  InputTag
  ParameterSet::getUntrackedParameter<InputTag>(std::string const& name, InputTag const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getInputTag();
  }

  template<>
  inline
  InputTag
  ParameterSet::getUntrackedParameter<InputTag>(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getInputTag();
  }

  template<>
  inline
  std::vector<InputTag>
  ParameterSet::getUntrackedParameter<std::vector<InputTag> >(std::string const& name, 
                                      std::vector<InputTag> const& defaultValue) const 
  {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getVInputTag();
  }

  template<>
  inline
  std::vector<InputTag>
  ParameterSet::getUntrackedParameter<std::vector<InputTag> >(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getVInputTag();
  }

  // ----------------------------------------------------------------------
  // EventID, VEventID

  template<>
  inline
  EventID
  ParameterSet::getUntrackedParameter<EventID>(std::string const& name, EventID const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getEventID();
  }

  template<>
  inline
  EventID
  ParameterSet::getUntrackedParameter<EventID>(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getEventID();
  }

  template<>
  inline
  std::vector<EventID>
  ParameterSet::getUntrackedParameter<std::vector<EventID> >(std::string const& name,
                                      std::vector<EventID> const& defaultValue) const
  {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getVEventID();
  }

  template<>
  inline
  std::vector<EventID>
  ParameterSet::getUntrackedParameter<std::vector<EventID> >(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getVEventID();
  }

  // ----------------------------------------------------------------------
  // LuminosityBlockID, VLuminosityBlockID

  template<>
  inline
  LuminosityBlockID
  ParameterSet::getUntrackedParameter<LuminosityBlockID>(std::string const& name, LuminosityBlockID const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getLuminosityBlockID();
  }

  template<>
  inline
  LuminosityBlockID
  ParameterSet::getUntrackedParameter<LuminosityBlockID>(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getLuminosityBlockID();
  }

  template<>
  inline
  std::vector<LuminosityBlockID>
  ParameterSet::getUntrackedParameter<std::vector<LuminosityBlockID> >(std::string const& name,
                                      std::vector<LuminosityBlockID> const& defaultValue) const
  {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getVLuminosityBlockID();
  }

  template<>
  inline
  std::vector<LuminosityBlockID>
  ParameterSet::getUntrackedParameter<std::vector<LuminosityBlockID> >(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getVLuminosityBlockID();
  }

  
  // ----------------------------------------------------------------------
  // PSet, vPSet
  
  template<>
  inline 
  ParameterSet
  ParameterSet::getUntrackedParameter<ParameterSet>(std::string const& name, ParameterSet const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getPSet();
  }

  template<>
  inline
  ParameterSet
  ParameterSet::getUntrackedParameter<ParameterSet>(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getPSet();
  }

  template<>
  inline 
  std::vector<ParameterSet>
  ParameterSet::getUntrackedParameter<std::vector<ParameterSet> >(std::string const& name, std::vector<ParameterSet> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getVPSet();
  }

  template<>
  inline
  std::vector<ParameterSet>
  ParameterSet::getUntrackedParameter<std::vector<ParameterSet> >(std::string const& name) const {
    return getEntryPointerOrThrow_(name)->getVPSet();
  }

  // specializations
  // ----------------------------------------------------------------------
  // Bool, vBool
  
  template<>
  inline 
  bool
  ParameterSet::getParameter<bool>(char const* name) const {
    return retrieve(name).getBool();
  }

  // ----------------------------------------------------------------------
  // Int32, vInt32
  
  template<>
  inline 
  int
  ParameterSet::getParameter<int>(char const* name) const {
    return retrieve(name).getInt32();
  }

  template<>
  inline 
  std::vector<int>
  ParameterSet::getParameter<std::vector<int> >(char const* name) const {
    return retrieve(name).getVInt32();
  }
  
 // ----------------------------------------------------------------------
  // Int64, vInt64

  template<>
  inline
  boost::int64_t
  ParameterSet::getParameter<boost::int64_t>(char const* name) const {
    return retrieve(name).getInt64();
  }

  template<>
  inline
  std::vector<boost::int64_t>
  ParameterSet::getParameter<std::vector<boost::int64_t> >(char const* name) const {
    return retrieve(name).getVInt64();
  }

  // ----------------------------------------------------------------------
  // Uint32, vUint32
  
  template<>
  inline 
  unsigned int
  ParameterSet::getParameter<unsigned int>(char const* name) const {
    return retrieve(name).getUInt32();
  }
  
  template<>
  inline 
  std::vector<unsigned int>
  ParameterSet::getParameter<std::vector<unsigned int> >(char const* name) const {
    return retrieve(name).getVUInt32();
  }
  
  // ----------------------------------------------------------------------
  // Uint64, vUint64

  template<>
  inline
  boost::uint64_t
  ParameterSet::getParameter<boost::uint64_t>(char const* name) const {
    return retrieve(name).getUInt64();
  }

  template<>
  inline
  std::vector<boost::uint64_t>
  ParameterSet::getParameter<std::vector<boost::uint64_t> >(char const* name) const {
    return retrieve(name).getVUInt64();
  }

  // ----------------------------------------------------------------------
  // Double, vDouble
  
  template<>
  inline 
  double
  ParameterSet::getParameter<double>(char const* name) const {
    return retrieve(name).getDouble();
  }
  
  template<>
  inline 
  std::vector<double>
  ParameterSet::getParameter<std::vector<double> >(char const* name) const {
    return retrieve(name).getVDouble();
  }
  
  // ----------------------------------------------------------------------
  // String, vString
  
  template<>
  inline 
  std::string
  ParameterSet::getParameter<std::string>(char const* name) const {
    return retrieve(name).getString();
  }
  
  template<>
  inline 
  std::vector<std::string>
  ParameterSet::getParameter<std::vector<std::string> >(char const* name) const {
    return retrieve(name).getVString();
  }

  // ----------------------------------------------------------------------
  // FileInPath

  template <>
  inline
  FileInPath
  ParameterSet::getParameter<FileInPath>(char const* name) const {
    return retrieve(name).getFileInPath();
  }
  
  // ----------------------------------------------------------------------
  // InputTag

  template <>
  inline
  InputTag
  ParameterSet::getParameter<InputTag>(char const* name) const {
    Entry const& e_input = retrieve(name);
    switch (e_input.typeCode()) 
    {
      case 't':   // InputTag
        return e_input.getInputTag();
      case 'S':   // string
        std::string const& label = e_input.getString();
	depricatedInputTagWarning(name, label);
        return InputTag( label );
    }
    throw edm::Exception(errors::Configuration, "ValueError") << "type of " 
       << name << " is expected to be InputTag or string (deprecated)";

  }

  // ----------------------------------------------------------------------
  // VInputTag

  template <>
  inline
  std::vector<InputTag>
  ParameterSet::getParameter<std::vector<InputTag> >(char const* name) const {
    return retrieve(name).getVInputTag();
  }

  // ----------------------------------------------------------------------
  // EventID

  template <>
  inline
  EventID
  ParameterSet::getParameter<EventID>(char const* name) const {
    return retrieve(name).getEventID();
  }

  // ----------------------------------------------------------------------
  // VEventID

  template <>
  inline
  std::vector<EventID>
  ParameterSet::getParameter<std::vector<EventID> >(char const* name) const {
    return retrieve(name).getVEventID();
  }

  // ----------------------------------------------------------------------
  // LuminosityBlockID

  template <>
  inline
  LuminosityBlockID
  ParameterSet::getParameter<LuminosityBlockID>(char const* name) const {
    return retrieve(name).getLuminosityBlockID();
  }

  // ----------------------------------------------------------------------
  // VLuminosityBlockID

  template <>
  inline
  std::vector<LuminosityBlockID>
  ParameterSet::getParameter<std::vector<LuminosityBlockID> >(char const* name) const {
    return retrieve(name).getVLuminosityBlockID();
  }

  // ----------------------------------------------------------------------
  // PSet, vPSet
  
  template<>
  inline 
  ParameterSet
  ParameterSet::getParameter<ParameterSet>(char const* name) const {
    return retrieve(name).getPSet();
  }
  
  template<>
  inline 
  std::vector<ParameterSet>
  ParameterSet::getParameter<std::vector<ParameterSet> >(char const* name) const {
    return retrieve(name).getVPSet();
  }

  // untracked parameters
  
  // ----------------------------------------------------------------------
  // Bool, vBool
  
  template<>
  inline 
  bool
  ParameterSet::getUntrackedParameter<bool>(char const* name, bool const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getBool();
  }

  template<>
  inline
  bool
  ParameterSet::getUntrackedParameter<bool>(char const* name) const {
    return getEntryPointerOrThrow_(name)->getBool();
  }
  
  
  // ----------------------------------------------------------------------
  // Int32, vInt32
  
  template<>
  inline 
  int
  ParameterSet::getUntrackedParameter<int>(char const* name, int const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getInt32();
  }

  template<>
  inline
  int
  ParameterSet::getUntrackedParameter<int>(char const* name) const {
    return getEntryPointerOrThrow_(name)->getInt32();
  }

  template<>
  inline 
  std::vector<int>
  ParameterSet::getUntrackedParameter<std::vector<int> >(char const* name, std::vector<int> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getVInt32();
  }

  template<>
  inline
  std::vector<int>
  ParameterSet::getUntrackedParameter<std::vector<int> >(char const* name) const {
    return getEntryPointerOrThrow_(name)->getVInt32();
  }
  
  // ----------------------------------------------------------------------
  // Uint32, vUint32
  
  template<>
  inline 
  unsigned int
  ParameterSet::getUntrackedParameter<unsigned int>(char const* name, unsigned int const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getUInt32();
  }

  template<>
  inline
  unsigned int
  ParameterSet::getUntrackedParameter<unsigned int>(char const* name) const {
    return getEntryPointerOrThrow_(name)->getUInt32();
  }
  
  template<>
  inline 
  std::vector<unsigned int>
  ParameterSet::getUntrackedParameter<std::vector<unsigned int> >(char const* name, std::vector<unsigned int> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getVUInt32();
  }

  template<>
  inline
  std::vector<unsigned int>
  ParameterSet::getUntrackedParameter<std::vector<unsigned int> >(char const* name) const {
    return getEntryPointerOrThrow_(name)->getVUInt32();
  }


  // ----------------------------------------------------------------------
  // Uint64, vUint64

  template<>
  inline
  boost::uint64_t
  ParameterSet::getUntrackedParameter<boost::uint64_t>(char const* name, boost::uint64_t const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getUInt64();
  }

  template<>
  inline
  boost::uint64_t
  ParameterSet::getUntrackedParameter<boost::uint64_t>(char const* name) const {
    return getEntryPointerOrThrow_(name)->getUInt64();
  }

  template<>
  inline
  std::vector<boost::uint64_t>
  ParameterSet::getUntrackedParameter<std::vector<boost::uint64_t> >(char const* name, std::vector<boost::uint64_t> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getVUInt64();
  }

  template<>
  inline
  std::vector<boost::uint64_t>
  ParameterSet::getUntrackedParameter<std::vector<boost::uint64_t> >(char const* name) const {
    return getEntryPointerOrThrow_(name)->getVUInt64();
  }


  // ----------------------------------------------------------------------
  // Int64, Vint64

  template<>
  inline
  boost::int64_t
  ParameterSet::getUntrackedParameter<boost::int64_t>(char const* name, boost::int64_t const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getInt64();
  }

  template<>
  inline
  boost::int64_t
  ParameterSet::getUntrackedParameter<boost::int64_t>(char const* name) const {
    return getEntryPointerOrThrow_(name)->getInt64();
  }

  template<>
  inline
  std::vector<boost::int64_t>
  ParameterSet::getUntrackedParameter<std::vector<boost::int64_t> >(char const* name, std::vector<boost::int64_t> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getVInt64();
  }

  template<>
  inline
  std::vector<boost::int64_t>
  ParameterSet::getUntrackedParameter<std::vector<boost::int64_t> >(char const* name) const {
    return getEntryPointerOrThrow_(name)->getVInt64();
  }

  
  // ----------------------------------------------------------------------
  // Double, vDouble
  
  template<>
  inline 
  double
  ParameterSet::getUntrackedParameter<double>(char const* name, double const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getDouble();
  }


  template<>
  inline
  double
  ParameterSet::getUntrackedParameter<double>(char const* name) const {
    return getEntryPointerOrThrow_(name)->getDouble();
  }  
  
  template<>
  inline 
  std::vector<double>
  ParameterSet::getUntrackedParameter<std::vector<double> >(char const* name, std::vector<double> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name); return entryPtr == 0 ? defaultValue : entryPtr->getVDouble(); 
  }

  template<>
  inline
  std::vector<double>
  ParameterSet::getUntrackedParameter<std::vector<double> >(char const* name) const {
    return getEntryPointerOrThrow_(name)->getVDouble();
  }
  
  // ----------------------------------------------------------------------
  // String, vString
  
  template<>
  inline 
  std::string
  ParameterSet::getUntrackedParameter<std::string>(char const* name, std::string const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getString();
  }

  template<>
  inline
  std::string
  ParameterSet::getUntrackedParameter<std::string>(char const* name) const {
    return getEntryPointerOrThrow_(name)->getString();
  }
  
  template<>
  inline 
  std::vector<std::string>
  ParameterSet::getUntrackedParameter<std::vector<std::string> >(char const* name, std::vector<std::string> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getVString();
  }


  template<>
  inline
  std::vector<std::string>
  ParameterSet::getUntrackedParameter<std::vector<std::string> >(char const* name) const {
    return getEntryPointerOrThrow_(name)->getVString();
  }

  // ----------------------------------------------------------------------
  //  FileInPath

  template<>
  inline
  FileInPath
  ParameterSet::getUntrackedParameter<FileInPath>(char const* name, FileInPath const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getFileInPath();
  }

  template<>
  inline
  FileInPath
  ParameterSet::getUntrackedParameter<FileInPath>(char const* name) const {
    return getEntryPointerOrThrow_(name)->getFileInPath();
  }

  // ----------------------------------------------------------------------
  // InputTag, VInputTag

  template<>
  inline
  InputTag
  ParameterSet::getUntrackedParameter<InputTag>(char const* name, InputTag const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getInputTag();
  }

  template<>
  inline
  InputTag
  ParameterSet::getUntrackedParameter<InputTag>(char const* name) const {
    return getEntryPointerOrThrow_(name)->getInputTag();
  }

  template<>
  inline
  std::vector<InputTag>
  ParameterSet::getUntrackedParameter<std::vector<InputTag> >(char const* name, 
                                      std::vector<InputTag> const& defaultValue) const 
  {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getVInputTag();
  }

  template<>
  inline
  std::vector<InputTag>
  ParameterSet::getUntrackedParameter<std::vector<InputTag> >(char const* name) const {
    return getEntryPointerOrThrow_(name)->getVInputTag();
  }

  // ----------------------------------------------------------------------
  // EventID, VEventID

  template<>
  inline
  EventID
  ParameterSet::getUntrackedParameter<EventID>(char const* name, EventID const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getEventID();
  }

  template<>
  inline
  EventID
  ParameterSet::getUntrackedParameter<EventID>(char const* name) const {
    return getEntryPointerOrThrow_(name)->getEventID();
  }

  template<>
  inline
  std::vector<EventID>
  ParameterSet::getUntrackedParameter<std::vector<EventID> >(char const* name,
                                      std::vector<EventID> const& defaultValue) const
  {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getVEventID();
  }

  template<>
  inline
  std::vector<EventID>
  ParameterSet::getUntrackedParameter<std::vector<EventID> >(char const* name) const {
    return getEntryPointerOrThrow_(name)->getVEventID();
  }

  // ----------------------------------------------------------------------
  // LuminosityBlockID, VLuminosityBlockID

  template<>
  inline
  LuminosityBlockID
  ParameterSet::getUntrackedParameter<LuminosityBlockID>(char const* name, LuminosityBlockID const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getLuminosityBlockID();
  }

  template<>
  inline
  LuminosityBlockID
  ParameterSet::getUntrackedParameter<LuminosityBlockID>(char const* name) const {
    return getEntryPointerOrThrow_(name)->getLuminosityBlockID();
  }

  template<>
  inline
  std::vector<LuminosityBlockID>
  ParameterSet::getUntrackedParameter<std::vector<LuminosityBlockID> >(char const* name,
                                      std::vector<LuminosityBlockID> const& defaultValue) const
  {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getVLuminosityBlockID();
  }

  template<>
  inline
  std::vector<LuminosityBlockID>
  ParameterSet::getUntrackedParameter<std::vector<LuminosityBlockID> >(char const* name) const {
    return getEntryPointerOrThrow_(name)->getVLuminosityBlockID();
  }

  
  // ----------------------------------------------------------------------
  // PSet, vPSet
  
  template<>
  inline 
  ParameterSet
  ParameterSet::getUntrackedParameter<ParameterSet>(char const* name, ParameterSet const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getPSet();
  }

  template<>
  inline
  ParameterSet
  ParameterSet::getUntrackedParameter<ParameterSet>(char const* name) const {
    return getEntryPointerOrThrow_(name)->getPSet();
  }

  template<>
  inline 
  std::vector<ParameterSet>
  ParameterSet::getUntrackedParameter<std::vector<ParameterSet> >(char const* name, std::vector<ParameterSet> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getVPSet();
  }

  template<>
  inline
  std::vector<ParameterSet>
  ParameterSet::getUntrackedParameter<std::vector<ParameterSet> >(char const* name) const {
    return getEntryPointerOrThrow_(name)->getVPSet();
  }

  // Associated functions used elsewhere in the ParameterSet system
  namespace pset
  {
    // Put into 'results' each parameter set in 'top', including 'top'
    // itself.
    void explode(ParameterSet const& top,
	       std::vector<ParameterSet>& results);
  }

  // Free function to retrieve a parameter set, given the parameter set ID.
  ParameterSet
  getParameterSet(ParameterSetID const& id);

}  // namespace edm
#endif

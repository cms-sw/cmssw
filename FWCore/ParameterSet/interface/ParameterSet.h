#ifndef FWCore_ParameterSet_ParameterSet_h
#define FWCore_ParameterSet_ParameterSet_h

// ----------------------------------------------------------------------
// Declaration for ParameterSet(parameter set) and related types
// ----------------------------------------------------------------------

// ----------------------------------------------------------------------
// prolog

// ----------------------------------------------------------------------
// prerequisite source files and headers

#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "FWCore/ParameterSet/interface/Entry.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSetEntry.h"
#include "FWCore/ParameterSet/interface/VParameterSetEntry.h"

#include <iosfwd>
#include <map>
#include <memory>
#include <string>
#include <vector>

// ----------------------------------------------------------------------
// contents
namespace cms {
  class Digest;
}

namespace edm {
  typedef std::vector<ParameterSet> VParameterSet;

  class ParameterSet {
  public:
    enum Bool {
      False = 0,
      True = 1,
      Unknown = 2
    };

    // default-construct
    ParameterSet();

    // construct from coded string.
    explicit ParameterSet(std::string const& rep);

    // construct from coded string and id.  Will cause registration
    ParameterSet(std::string const& rep, ParameterSetID const& id);

    ~ParameterSet();

    // instantiate in this library, so these methods don't cause code bloat
    ParameterSet(ParameterSet const& other);

    ParameterSet& operator=(ParameterSet const& other);

    void swap(ParameterSet& other);

    void copyForModify(ParameterSet const& other);

    // identification
    ParameterSetID id() const;
    void setID(ParameterSetID const& id);
    bool isRegistered() const {return id_.isValid();}
    ParameterSetID trackedID() const {return id();} // to be phased out.

    // Entry-handling
    Entry const& retrieve(char const*) const;
    Entry const& retrieve(std::string const&) const;
    Entry const* retrieveUntracked(char const*) const;
    Entry const* retrieveUntracked(std::string const&) const;
    Entry const* retrieveUnknown(char const*) const;
    Entry const* retrieveUnknown(std::string const&) const;
    ParameterSetEntry const& retrieveParameterSet(std::string const&) const;
    ParameterSetEntry const* retrieveUntrackedParameterSet(std::string const&) const;
    ParameterSetEntry const* retrieveUnknownParameterSet(std::string const&) const;
    VParameterSetEntry const& retrieveVParameterSet(std::string const&) const;
    VParameterSetEntry const* retrieveUntrackedVParameterSet(std::string const&) const;
    VParameterSetEntry const* retrieveUnknownVParameterSet(std::string const&) const;

    void insertParameterSet(bool okay_to_replace, std::string const& name, ParameterSetEntry const& entry);
    void insertVParameterSet(bool okay_to_replace, std::string const& name, VParameterSetEntry const& entry);
    void insert(bool ok_to_replace, char const* , Entry const&);
    void insert(bool ok_to_replace, std::string const&, Entry const&);
    void augment(ParameterSet const& from);
    void copyFrom(ParameterSet const& from, std::string const& name);
    std::string getParameterAsString(std::string const& name) const;

    // encode only tracked parameters
    std::string toString() const;
    void toString(std::string& result) const;
    void toDigest(cms::Digest &digest) const;

    //encode tracked and untracked parameters
    void allToString(std::string& result) const;

    template<typename T>
    T
    getParameter(std::string const&) const;

    template<typename T>
    T
    getParameter(char const*) const;

    ParameterSet const&
    getParameterSet(std::string const&) const;

    ParameterSet const&
    getParameterSet(char const*) const;

    ParameterSet const&
    getUntrackedParameterSet(std::string const& name, ParameterSet const& defaultValue) const;

    ParameterSet const&
    getUntrackedParameterSet(char const* name, ParameterSet const& defaultValue) const;

    ParameterSet const&
    getUntrackedParameterSet(std::string const& name) const;

    ParameterSet const&
    getUntrackedParameterSet(char const* name) const;

    VParameterSet const&
    getParameterSetVector(std::string const& name) const;

    VParameterSet const&
    getParameterSetVector(char const* name) const;

    VParameterSet const&
    getUntrackedParameterSetVector(std::string const& name, VParameterSet const& defaultValue) const;

    VParameterSet const&
    getUntrackedParameterSetVector(char const* name, VParameterSet const& defaultValue) const;

    VParameterSet const&
    getUntrackedParameterSetVector(std::string const& name) const;

    VParameterSet const&
    getUntrackedParameterSetVector(char const* name) const;

    template<typename T>
    void
    addParameter(std::string const& name, T const& value) {
      invalidateRegistration(name);
      insert(true, name, Entry(name, value, true));
    }

    template<typename T>
    void
    addParameter(char const* name, T const& value) {
      invalidateRegistration(name);
      insert(true, name, Entry(name, value, true));
    }

    template<typename T>
    T
    getUntrackedParameter(std::string const&, T const&) const;

    template<typename T>
    T
    getUntrackedParameter(char const*, T const&) const;

    template<typename T>
    T
    getUntrackedParameter(std::string const&) const;

    template<typename T>
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
    template<typename T>
    bool existsAs(std::string const& parameterName, bool trackiness=true) const {
       std::vector<std::string> names = getParameterNamesForType<T>(trackiness);
       return std::find(names.begin(), names.end(), parameterName) != names.end();
    }

    void deprecatedInputTagWarning(std::string const& name, std::string const& label) const;

    template<typename T>
    std::vector<std::string> getParameterNamesForType(bool trackiness = true) const {
      std::vector<std::string> result;
      // This is icky, but I don't know of another way in the current
      // code to get at the character code that denotes type T.
      T value = T();
      Entry type_translator("", value, trackiness);
      char type_code = type_translator.typeCode();

      (void)getNamesByCode_(type_code, trackiness, result);
      return result;
    }

    template<typename T>
    void
    addUntrackedParameter(std::string const& name, T const& value) {
      insert(true, name, Entry(name, value, false));
    }

    template<typename T>
    void
    addUntrackedParameter(char const* name, T const& value) {
      insert(true, name, Entry(name, value, false));
    }

    bool empty() const {
      return tbl_.empty() && psetTable_.empty() && vpsetTable_.empty();
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
    std::string dump(unsigned int indent = 0) const;

    friend std::ostream& operator << (std::ostream& os, ParameterSet const& pset);

    ParameterSet const& registerIt();

    std::auto_ptr<ParameterSet> popParameterSet(std::string const& name);
    void eraseSimpleParameter(std::string const& name);
    void eraseOrSetUntrackedParameterSet(std::string const& name);

    std::auto_ptr<std::vector<ParameterSet> > popVParameterSet(std::string const& name);

    typedef std::map<std::string, Entry> table;
    table const& tbl() const {return tbl_;}

    typedef std::map<std::string, ParameterSetEntry> psettable;
    psettable const& psetTable() const {return psetTable_;}

    typedef std::map<std::string, VParameterSetEntry> vpsettable;
    vpsettable const& vpsetTable() const {return vpsetTable_;}

    ParameterSet*
    getPSetForUpdate(std::string const& name, bool& isTracked);

    ParameterSet*
    getPSetForUpdate(std::string const& name) {
      bool isTracked = false;
      return getPSetForUpdate(name, isTracked);
    }

    VParameterSetEntry*
    getPSetVectorForUpdate(std::string const& name);

  private:
    // decode
    bool fromString(std::string const&);

    void toStringImp(std::string&, bool useAll) const;

    table tbl_;
    psettable psetTable_;
    vpsettable vpsetTable_;

    // If the id_ is invalid, that means a new value should be
    // calculated before the value is returned. Upon registration, the
    // id_ is made valid. Updating any tracked parameter invalidates the id_.
    ParameterSetID id_;

    void invalidateRegistration(std::string const& nameOfTracked);

    void calculateID();

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
  void swap (ParameterSet& a, ParameterSet& b) {
    a.swap(b);
  }

  bool operator==(ParameterSet const& a, ParameterSet const& b);

  bool isTransientEqual(ParameterSet const& a, ParameterSet const& b);

  inline
  bool
  operator!=(ParameterSet const& a, ParameterSet const& b) {
    return !(a == b);
  }

  // Free function to retrieve a parameter set, given the parameter set ID.
  ParameterSet const&
  getParameterSet(ParameterSetID const& id);

  // specializations
  // ----------------------------------------------------------------------

  template<>
  bool
  ParameterSet::getParameter<bool>(std::string const& name) const;

  // ----------------------------------------------------------------------
  // Int32, vInt32

  template<>
  int
  ParameterSet::getParameter<int>(std::string const& name) const;

  template<>
  std::vector<int>
  ParameterSet::getParameter<std::vector<int> >(std::string const& name) const;

 // ----------------------------------------------------------------------
  // Int64, vInt64

  template<>
  long long
  ParameterSet::getParameter<long long>(std::string const& name) const;

  template<>
  std::vector<long long>
  ParameterSet::getParameter<std::vector<long long> >(std::string const& name) const;

  // ----------------------------------------------------------------------
  // Uint32, vUint32

  template<>
  unsigned int
  ParameterSet::getParameter<unsigned int>(std::string const& name) const;

  template<>
  std::vector<unsigned int>
  ParameterSet::getParameter<std::vector<unsigned int> >(std::string const& name) const;

  // ----------------------------------------------------------------------
  // Uint64, vUint64

  template<>
  unsigned long long
  ParameterSet::getParameter<unsigned long long>(std::string const& name) const;

  template<>
  std::vector<unsigned long long>
  ParameterSet::getParameter<std::vector<unsigned long long> >(std::string const& name) const;

  // ----------------------------------------------------------------------
  // Double, vDouble

  template<>
  double
  ParameterSet::getParameter<double>(std::string const& name) const;

  template<>
  std::vector<double>
  ParameterSet::getParameter<std::vector<double> >(std::string const& name) const;

  // ----------------------------------------------------------------------
  // String, vString

  template<>
  std::string
  ParameterSet::getParameter<std::string>(std::string const& name) const;

  template<>
  std::vector<std::string>
  ParameterSet::getParameter<std::vector<std::string> >(std::string const& name) const;

  // ----------------------------------------------------------------------
  // FileInPath

  template<>
  FileInPath
  ParameterSet::getParameter<FileInPath>(std::string const& name) const;

  // FileInPath can't default-construct something useful, so we specialize
  // this template
  template<>
  std::vector<std::string>
  ParameterSet::getParameterNamesForType<FileInPath>(bool trackiness) const;

  // ----------------------------------------------------------------------
  // InputTag

  template<>
  InputTag
  ParameterSet::getParameter<InputTag>(std::string const& name) const;

  // ----------------------------------------------------------------------
  // VInputTag

  template<>
  std::vector<InputTag>
  ParameterSet::getParameter<std::vector<InputTag> >(std::string const& name) const;

   // ----------------------------------------------------------------------
   // ESInputTag

   template<>
   ESInputTag
   ParameterSet::getParameter<ESInputTag>(std::string const& name) const;

   // ----------------------------------------------------------------------
   // VESInputTag

   template<>
   std::vector<ESInputTag>
   ParameterSet::getParameter<std::vector<ESInputTag> >(std::string const& name) const;

  // ----------------------------------------------------------------------
  // EventID

  template<>
  EventID
  ParameterSet::getParameter<EventID>(std::string const& name) const;

  // ----------------------------------------------------------------------
  // VEventID

  template<>
  std::vector<EventID>
  ParameterSet::getParameter<std::vector<EventID> >(std::string const& name) const;

  // ----------------------------------------------------------------------
  // LuminosityBlockID

  template<>
  LuminosityBlockID
  ParameterSet::getParameter<LuminosityBlockID>(std::string const& name) const;

  // ----------------------------------------------------------------------
  // VLuminosityBlockID

  template<>
  std::vector<LuminosityBlockID>
  ParameterSet::getParameter<std::vector<LuminosityBlockID> >(std::string const& name) const;

  // ----------------------------------------------------------------------
  // EventRange

  template<>
  EventRange
  ParameterSet::getParameter<EventRange>(std::string const& name) const;

  // ----------------------------------------------------------------------
  // VEventRange

  template<>
  std::vector<EventRange>
  ParameterSet::getParameter<std::vector<EventRange> >(std::string const& name) const;

  // ----------------------------------------------------------------------
  // LuminosityBlockRange

  template<>
  LuminosityBlockRange
  ParameterSet::getParameter<LuminosityBlockRange>(std::string const& name) const;

  // ----------------------------------------------------------------------
  // VLuminosityBlockRange

  template<>
  std::vector<LuminosityBlockRange>
  ParameterSet::getParameter<std::vector<LuminosityBlockRange> >(std::string const& name) const;

  // ----------------------------------------------------------------------
  // PSet, vPSet

  template<>
  ParameterSet
  ParameterSet::getParameter<ParameterSet>(std::string const& name) const;

  template<>
  VParameterSet
  ParameterSet::getParameter<VParameterSet>(std::string const& name) const;

  template<>
  void
  ParameterSet::addParameter<ParameterSet>(std::string const& name, ParameterSet const& value);

  template<>
  void
  ParameterSet::addParameter<ParameterSet>(char const* name, ParameterSet const& value);

  template<>
  void
  ParameterSet::addUntrackedParameter<ParameterSet>(std::string const& name, ParameterSet const& value);

  template<>
  void
  ParameterSet::addUntrackedParameter<ParameterSet>(char const* name, ParameterSet const& value);

  template<>
  void
  ParameterSet::addParameter<VParameterSet>(std::string const& name, VParameterSet const& value);

  template<>
  void
  ParameterSet::addParameter<VParameterSet>(char const* name, VParameterSet const& value);

  template<>
  void
  ParameterSet::addUntrackedParameter<VParameterSet>(std::string const& name, VParameterSet const& value);

  template<>
  void
  ParameterSet::addUntrackedParameter<VParameterSet>(char const* name, VParameterSet const& value);

  // untracked parameters

  // ----------------------------------------------------------------------
  // Bool, vBool

  template<>
  bool
  ParameterSet::getUntrackedParameter<bool>(std::string const& name, bool const& defaultValue) const;

  template<>
  bool
  ParameterSet::getUntrackedParameter<bool>(std::string const& name) const;

  // ----------------------------------------------------------------------
  // Int32, vInt32

  template<>
  int
  ParameterSet::getUntrackedParameter<int>(std::string const& name, int const& defaultValue) const;

  template<>
  int
  ParameterSet::getUntrackedParameter<int>(std::string const& name) const;

  template<>
  std::vector<int>
  ParameterSet::getUntrackedParameter<std::vector<int> >(std::string const& name, std::vector<int> const& defaultValue) const;

  template<>
  std::vector<int>
  ParameterSet::getUntrackedParameter<std::vector<int> >(std::string const& name) const;

  // ----------------------------------------------------------------------
  // Uint32, vUint32

  template<>
  unsigned int
  ParameterSet::getUntrackedParameter<unsigned int>(std::string const& name, unsigned int const& defaultValue) const;

  template<>
  unsigned int
  ParameterSet::getUntrackedParameter<unsigned int>(std::string const& name) const;

  template<>
  std::vector<unsigned int>
  ParameterSet::getUntrackedParameter<std::vector<unsigned int> >(std::string const& name, std::vector<unsigned int> const& defaultValue) const;

  template<>
  std::vector<unsigned int>
  ParameterSet::getUntrackedParameter<std::vector<unsigned int> >(std::string const& name) const;

  // ----------------------------------------------------------------------
  // Uint64, vUint64

  template<>
  unsigned long long
  ParameterSet::getUntrackedParameter<unsigned long long>(std::string const& name, unsigned long long const& defaultValue) const;

  template<>
  unsigned long long
  ParameterSet::getUntrackedParameter<unsigned long long>(std::string const& name) const;

  template<>
  std::vector<unsigned long long>
  ParameterSet::getUntrackedParameter<std::vector<unsigned long long> >(std::string const& name, std::vector<unsigned long long> const& defaultValue) const;

  template<>
  std::vector<unsigned long long>
  ParameterSet::getUntrackedParameter<std::vector<unsigned long long> >(std::string const& name) const;

  // ----------------------------------------------------------------------
  // Int64, Vint64

  template<>
  long long
  ParameterSet::getUntrackedParameter<long long>(std::string const& name, long long const& defaultValue) const;

  template<>
  long long
  ParameterSet::getUntrackedParameter<long long>(std::string const& name) const;

  template<>
  std::vector<long long>
  ParameterSet::getUntrackedParameter<std::vector<long long> >(std::string const& name, std::vector<long long> const& defaultValue) const;

  template<>
  std::vector<long long>
  ParameterSet::getUntrackedParameter<std::vector<long long> >(std::string const& name) const;

  // ----------------------------------------------------------------------
  // Double, vDouble

  template<>
  double
  ParameterSet::getUntrackedParameter<double>(std::string const& name, double const& defaultValue) const;

  template<>
  double
  ParameterSet::getUntrackedParameter<double>(std::string const& name) const;

  template<>
  std::vector<double>
  ParameterSet::getUntrackedParameter<std::vector<double> >(std::string const& name, std::vector<double> const& defaultValue) const;

  template<>
  std::vector<double>
  ParameterSet::getUntrackedParameter<std::vector<double> >(std::string const& name) const;

  // ----------------------------------------------------------------------
  // String, vString

  template<>
  std::string
  ParameterSet::getUntrackedParameter<std::string>(std::string const& name, std::string const& defaultValue) const;

  template<>
  std::string
  ParameterSet::getUntrackedParameter<std::string>(std::string const& name) const;

  template<>
  std::vector<std::string>
  ParameterSet::getUntrackedParameter<std::vector<std::string> >(std::string const& name, std::vector<std::string> const& defaultValue) const;

  template<>
  std::vector<std::string>
  ParameterSet::getUntrackedParameter<std::vector<std::string> >(std::string const& name) const;

  // ----------------------------------------------------------------------
  //  FileInPath

  template<>
  FileInPath
  ParameterSet::getUntrackedParameter<FileInPath>(std::string const& name, FileInPath const& defaultValue) const;

  template<>
  FileInPath
  ParameterSet::getUntrackedParameter<FileInPath>(std::string const& name) const;

  // ----------------------------------------------------------------------
  // InputTag, VInputTag

  template<>
  InputTag
  ParameterSet::getUntrackedParameter<InputTag>(std::string const& name, InputTag const& defaultValue) const;

  template<>
  InputTag
  ParameterSet::getUntrackedParameter<InputTag>(std::string const& name) const;

  template<>
  std::vector<InputTag>
  ParameterSet::getUntrackedParameter<std::vector<InputTag> >(std::string const& name,
                                      std::vector<InputTag> const& defaultValue) const;

  template<>
  std::vector<InputTag>
  ParameterSet::getUntrackedParameter<std::vector<InputTag> >(std::string const& name) const;

  // ----------------------------------------------------------------------
  // EventID, VEventID

  template<>
  EventID
  ParameterSet::getUntrackedParameter<EventID>(std::string const& name, EventID const& defaultValue) const;

  template<>
  EventID
  ParameterSet::getUntrackedParameter<EventID>(std::string const& name) const;

  template<>
  std::vector<EventID>
  ParameterSet::getUntrackedParameter<std::vector<EventID> >(std::string const& name,
                                      std::vector<EventID> const& defaultValue) const;
  template<>
  std::vector<EventID>
  ParameterSet::getUntrackedParameter<std::vector<EventID> >(std::string const& name) const;

  // ----------------------------------------------------------------------
  // LuminosityBlockID, VLuminosityBlockID

  template<>
  LuminosityBlockID
  ParameterSet::getUntrackedParameter<LuminosityBlockID>(std::string const& name, LuminosityBlockID const& defaultValue) const;

  template<>
  LuminosityBlockID
  ParameterSet::getUntrackedParameter<LuminosityBlockID>(std::string const& name) const;

  template<>
  std::vector<LuminosityBlockID>
  ParameterSet::getUntrackedParameter<std::vector<LuminosityBlockID> >(std::string const& name,
                                      std::vector<LuminosityBlockID> const& defaultValue) const;
  template<>
  std::vector<LuminosityBlockID>
  ParameterSet::getUntrackedParameter<std::vector<LuminosityBlockID> >(std::string const& name) const;

  // ----------------------------------------------------------------------
  // EventRange, VEventRange

  template<>
  EventRange
  ParameterSet::getUntrackedParameter<EventRange>(std::string const& name, EventRange const& defaultValue) const;

  template<>
  EventRange
  ParameterSet::getUntrackedParameter<EventRange>(std::string const& name) const;

  template<>
  std::vector<EventRange>
  ParameterSet::getUntrackedParameter<std::vector<EventRange> >(std::string const& name,
                                      std::vector<EventRange> const& defaultValue) const;
  template<>
  std::vector<EventRange>
  ParameterSet::getUntrackedParameter<std::vector<EventRange> >(std::string const& name) const;

  // ----------------------------------------------------------------------
  // LuminosityBlockRange, VLuminosityBlockRange

  template<>
  LuminosityBlockRange
  ParameterSet::getUntrackedParameter<LuminosityBlockRange>(std::string const& name, LuminosityBlockRange const& defaultValue) const;

  template<>
  LuminosityBlockRange
  ParameterSet::getUntrackedParameter<LuminosityBlockRange>(std::string const& name) const;

  template<>
  std::vector<LuminosityBlockRange>
  ParameterSet::getUntrackedParameter<std::vector<LuminosityBlockRange> >(std::string const& name,
                                      std::vector<LuminosityBlockRange> const& defaultValue) const;
  template<>
  std::vector<LuminosityBlockRange>
  ParameterSet::getUntrackedParameter<std::vector<LuminosityBlockRange> >(std::string const& name) const;

  // specializations
  // ----------------------------------------------------------------------
  // Bool, vBool

  template<>
  bool
  ParameterSet::getParameter<bool>(char const* name) const;

  // ----------------------------------------------------------------------
  // Int32, vInt32

  template<>
  int
  ParameterSet::getParameter<int>(char const* name) const;

  template<>
  std::vector<int>
  ParameterSet::getParameter<std::vector<int> >(char const* name) const;

 // ----------------------------------------------------------------------
  // Int64, vInt64

  template<>
  long long
  ParameterSet::getParameter<long long>(char const* name) const;

  template<>
  std::vector<long long>
  ParameterSet::getParameter<std::vector<long long> >(char const* name) const;

  // ----------------------------------------------------------------------
  // Uint32, vUint32

  template<>
  unsigned int
  ParameterSet::getParameter<unsigned int>(char const* name) const;

  template<>
  std::vector<unsigned int>
  ParameterSet::getParameter<std::vector<unsigned int> >(char const* name) const;

  // ----------------------------------------------------------------------
  // Uint64, vUint64

  template<>
  unsigned long long
  ParameterSet::getParameter<unsigned long long>(char const* name) const;

  template<>
  std::vector<unsigned long long>
  ParameterSet::getParameter<std::vector<unsigned long long> >(char const* name) const;

  // ----------------------------------------------------------------------
  // Double, vDouble

  template<>
  double
  ParameterSet::getParameter<double>(char const* name) const;

  template<>
  std::vector<double>
  ParameterSet::getParameter<std::vector<double> >(char const* name) const;

  // ----------------------------------------------------------------------
  // String, vString

  template<>
  std::string
  ParameterSet::getParameter<std::string>(char const* name) const;

  template<>
  std::vector<std::string>
  ParameterSet::getParameter<std::vector<std::string> >(char const* name) const;

  // ----------------------------------------------------------------------
  // FileInPath

  template<>
  FileInPath
  ParameterSet::getParameter<FileInPath>(char const* name) const;

  // ----------------------------------------------------------------------
  // InputTag

  template<>
  InputTag
  ParameterSet::getParameter<InputTag>(char const* name) const;

  // ----------------------------------------------------------------------
  // VInputTag

  template<>
  std::vector<InputTag>
  ParameterSet::getParameter<std::vector<InputTag> >(char const* name) const;

  // ----------------------------------------------------------------------
  // EventID

  template<>
  EventID
  ParameterSet::getParameter<EventID>(char const* name) const;

  // ----------------------------------------------------------------------
  // VEventID

  template<>
  std::vector<EventID>
  ParameterSet::getParameter<std::vector<EventID> >(char const* name) const;

  // ----------------------------------------------------------------------
  // LuminosityBlockID

  template<>
  LuminosityBlockID
  ParameterSet::getParameter<LuminosityBlockID>(char const* name) const;

  // ----------------------------------------------------------------------
  // VLuminosityBlockID

  template<>
  std::vector<LuminosityBlockID>
  ParameterSet::getParameter<std::vector<LuminosityBlockID> >(char const* name) const;

  // ----------------------------------------------------------------------
  // EventRange

  template<>
  EventRange
  ParameterSet::getParameter<EventRange>(char const* name) const;

  // ----------------------------------------------------------------------
  // VEventRange

  template<>
  std::vector<EventRange>
  ParameterSet::getParameter<std::vector<EventRange> >(char const* name) const;

  // ----------------------------------------------------------------------
  // LuminosityBlockRange

  template<>
  LuminosityBlockRange
  ParameterSet::getParameter<LuminosityBlockRange>(char const* name) const;

  // ----------------------------------------------------------------------
  // VLuminosityBlockRange

  template<>
  std::vector<LuminosityBlockRange>
  ParameterSet::getParameter<std::vector<LuminosityBlockRange> >(char const* name) const;

  // ----------------------------------------------------------------------
  // PSet, vPSet

  template<>
  ParameterSet
  ParameterSet::getParameter<ParameterSet>(char const* name) const;

  template<>
  VParameterSet
  ParameterSet::getParameter<VParameterSet>(char const* name) const;

  // untracked parameters

  // ----------------------------------------------------------------------
  // Bool, vBool

  template<>
  bool
  ParameterSet::getUntrackedParameter<bool>(char const* name, bool const& defaultValue) const;

  template<>
  bool
  ParameterSet::getUntrackedParameter<bool>(char const* name) const;

  // ----------------------------------------------------------------------
  // Int32, vInt32

  template<>
  int
  ParameterSet::getUntrackedParameter<int>(char const* name, int const& defaultValue) const;

  template<>
  int
  ParameterSet::getUntrackedParameter<int>(char const* name) const;

  template<>
  std::vector<int>
  ParameterSet::getUntrackedParameter<std::vector<int> >(char const* name, std::vector<int> const& defaultValue) const;

  template<>
  std::vector<int>
  ParameterSet::getUntrackedParameter<std::vector<int> >(char const* name) const;

  // ----------------------------------------------------------------------
  // Uint32, vUint32

  template<>
  unsigned int
  ParameterSet::getUntrackedParameter<unsigned int>(char const* name, unsigned int const& defaultValue) const;

  template<>
  unsigned int
  ParameterSet::getUntrackedParameter<unsigned int>(char const* name) const;

  template<>
  std::vector<unsigned int>
  ParameterSet::getUntrackedParameter<std::vector<unsigned int> >(char const* name, std::vector<unsigned int> const& defaultValue) const;

  template<>
  std::vector<unsigned int>
  ParameterSet::getUntrackedParameter<std::vector<unsigned int> >(char const* name) const;

  // ----------------------------------------------------------------------
  // Uint64, vUint64

  template<>
  unsigned long long
  ParameterSet::getUntrackedParameter<unsigned long long>(char const* name, unsigned long long const& defaultValue) const;

  template<>
  unsigned long long
  ParameterSet::getUntrackedParameter<unsigned long long>(char const* name) const;

  template<>
  std::vector<unsigned long long>
  ParameterSet::getUntrackedParameter<std::vector<unsigned long long> >(char const* name, std::vector<unsigned long long> const& defaultValue) const;

  template<>
  std::vector<unsigned long long>
  ParameterSet::getUntrackedParameter<std::vector<unsigned long long> >(char const* name) const;

  // ----------------------------------------------------------------------
  // Int64, Vint64

  template<>
  long long
  ParameterSet::getUntrackedParameter<long long>(char const* name, long long const& defaultValue) const;

  template<>
  long long
  ParameterSet::getUntrackedParameter<long long>(char const* name) const;

  template<>
  std::vector<long long>
  ParameterSet::getUntrackedParameter<std::vector<long long> >(char const* name, std::vector<long long> const& defaultValue) const;

  template<>
  std::vector<long long>
  ParameterSet::getUntrackedParameter<std::vector<long long> >(char const* name) const;

  // ----------------------------------------------------------------------
  // Double, vDouble

  template<>
  double
  ParameterSet::getUntrackedParameter<double>(char const* name, double const& defaultValue) const;

  template<>
  double
  ParameterSet::getUntrackedParameter<double>(char const* name) const;

  template<>
  std::vector<double>
  ParameterSet::getUntrackedParameter<std::vector<double> >(char const* name, std::vector<double> const& defaultValue) const;

  template<>
  std::vector<double>
  ParameterSet::getUntrackedParameter<std::vector<double> >(char const* name) const;

  // ----------------------------------------------------------------------
  // String, vString

  template<>
  std::string
  ParameterSet::getUntrackedParameter<std::string>(char const* name, std::string const& defaultValue) const;

  template<>
  std::string
  ParameterSet::getUntrackedParameter<std::string>(char const* name) const;

  template<>
  std::vector<std::string>
  ParameterSet::getUntrackedParameter<std::vector<std::string> >(char const* name, std::vector<std::string> const& defaultValue) const;

  template<>
  std::vector<std::string>
  ParameterSet::getUntrackedParameter<std::vector<std::string> >(char const* name) const;

  // ----------------------------------------------------------------------
  //  FileInPath

  template<>
  FileInPath
  ParameterSet::getUntrackedParameter<FileInPath>(char const* name, FileInPath const& defaultValue) const;

  template<>
  FileInPath
  ParameterSet::getUntrackedParameter<FileInPath>(char const* name) const;

  // ----------------------------------------------------------------------
  // InputTag, VInputTag

  template<>
  InputTag
  ParameterSet::getUntrackedParameter<InputTag>(char const* name, InputTag const& defaultValue) const;

  template<>
  InputTag
  ParameterSet::getUntrackedParameter<InputTag>(char const* name) const;

  template<>
  std::vector<InputTag>
  ParameterSet::getUntrackedParameter<std::vector<InputTag> >(char const* name,
                                      std::vector<InputTag> const& defaultValue) const;

  template<>
  std::vector<InputTag>
  ParameterSet::getUntrackedParameter<std::vector<InputTag> >(char const* name) const;

  // ----------------------------------------------------------------------
  // EventID, VEventID

  template<>
  EventID
  ParameterSet::getUntrackedParameter<EventID>(char const* name, EventID const& defaultValue) const;

  template<>
  EventID
  ParameterSet::getUntrackedParameter<EventID>(char const* name) const;

  template<>
  std::vector<EventID>
  ParameterSet::getUntrackedParameter<std::vector<EventID> >(char const* name,
                                      std::vector<EventID> const& defaultValue) const;
  template<>
  std::vector<EventID>
  ParameterSet::getUntrackedParameter<std::vector<EventID> >(char const* name) const;

  // ----------------------------------------------------------------------
  // LuminosityBlockID, VLuminosityBlockID

  template<>
  LuminosityBlockID
  ParameterSet::getUntrackedParameter<LuminosityBlockID>(char const* name, LuminosityBlockID const& defaultValue) const;

  template<>
  LuminosityBlockID
  ParameterSet::getUntrackedParameter<LuminosityBlockID>(char const* name) const;

  template<>
  std::vector<LuminosityBlockID>
  ParameterSet::getUntrackedParameter<std::vector<LuminosityBlockID> >(char const* name,
                                      std::vector<LuminosityBlockID> const& defaultValue) const;
  template<>
  std::vector<LuminosityBlockID>
  ParameterSet::getUntrackedParameter<std::vector<LuminosityBlockID> >(char const* name) const;

  // ----------------------------------------------------------------------
  // EventRange, VEventRange

  template<>
  EventRange
  ParameterSet::getUntrackedParameter<EventRange>(char const* name, EventRange const& defaultValue) const;

  template<>
  EventRange
  ParameterSet::getUntrackedParameter<EventRange>(char const* name) const;

  template<>
  std::vector<EventRange>
  ParameterSet::getUntrackedParameter<std::vector<EventRange> >(char const* name,
                                      std::vector<EventRange> const& defaultValue) const;
  template<>
  std::vector<EventRange>
  ParameterSet::getUntrackedParameter<std::vector<EventRange> >(char const* name) const;

  // ----------------------------------------------------------------------
  // LuminosityBlockRange, VLuminosityBlockRange

  template<>
  LuminosityBlockRange
  ParameterSet::getUntrackedParameter<LuminosityBlockRange>(char const* name, LuminosityBlockRange const& defaultValue) const;

  template<>
  LuminosityBlockRange
  ParameterSet::getUntrackedParameter<LuminosityBlockRange>(char const* name) const;

  template<>
  std::vector<LuminosityBlockRange>
  ParameterSet::getUntrackedParameter<std::vector<LuminosityBlockRange> >(char const* name,
                                      std::vector<LuminosityBlockRange> const& defaultValue) const;
  template<>
  std::vector<LuminosityBlockRange>
  ParameterSet::getUntrackedParameter<std::vector<LuminosityBlockRange> >(char const* name) const;

  // ----------------------------------------------------------------------
  // PSet, vPSet

  template<>
  ParameterSet
  ParameterSet::getUntrackedParameter<ParameterSet>(char const* name, ParameterSet const& defaultValue) const;

  template<>
  ParameterSet
  ParameterSet::getUntrackedParameter<ParameterSet>(std::string const& name, ParameterSet const& defaultValue) const;

  template<>
  ParameterSet
  ParameterSet::getUntrackedParameter<ParameterSet>(char const* name) const;

  template<>
  ParameterSet
  ParameterSet::getUntrackedParameter<ParameterSet>(std::string const& name) const;

  template<>
  VParameterSet
  ParameterSet::getUntrackedParameter<VParameterSet>(char const* name, VParameterSet const& defaultValue) const;

  template<>
  VParameterSet
  ParameterSet::getUntrackedParameter<VParameterSet>(char const* name) const;

  template<>
  VParameterSet
  ParameterSet::getUntrackedParameter<VParameterSet>(std::string const& name, VParameterSet const& defaultValue) const;

  template<>
  VParameterSet
  ParameterSet::getUntrackedParameter<VParameterSet>(std::string const& name) const;

  template<>
  std::vector<std::string>
  ParameterSet::getParameterNamesForType<ParameterSet>(bool trackiness) const;

  template<>
  std::vector<std::string>
  ParameterSet::getParameterNamesForType<VParameterSet>(bool trackiness) const;

  ParameterSet::Bool
  operator&&(ParameterSet::Bool a, ParameterSet::Bool b);

}  // namespace edm
#endif

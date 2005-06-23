// ----------------------------------------------------------------------
// $Id: ParameterSet.h,v 1.5 2005/06/18 02:18:10 wmtan Exp $
//
// Declaration for ParameterSet(parameter set) and related types
// ----------------------------------------------------------------------


// ----------------------------------------------------------------------
// prolog

#ifndef  PARAMETERSET_H
#define  PARAMETERSET_H


// ----------------------------------------------------------------------
// prerequisite source files and headers

#include "FWCore/ParameterSet/interface/Entry.h"
#include <string>
#include <map>
#include <stdexcept>
#include <vector>


// ----------------------------------------------------------------------
// contents

namespace edm {

// ----------------------------------------------------------------------
// edm::ParameterSetError

  class ParameterSetError : public std::runtime_error {
  public:
    explicit ParameterSetError(std::string const& mesg) : std::runtime_error(mesg) {}

    virtual ~ParameterSetError() throw() {}

  };  // ParameterSetError


// ----------------------------------------------------------------------
// edm::ParameterSet

  class ParameterSet {
  public:
    // default-construct
    ParameterSet() : tbl() {}

    // construct from coded string
    explicit ParameterSet(std::string const&);

    // Entry-handling
    Entry const& retrieve(std::string const&) const;
    Entry const* const retrieveUntracked(std::string const&) const;
    void insert(bool ok_to_replace, std::string const& , Entry const&);
    void augment(ParameterSet const& from);

    // encode
    std::string toString() const;
    std::string toStringOfTracked() const;

    template< class T >
    T
    getParameter(std::string const&) const;

    template< class T >
    T
    getUntrackedParameter(std::string const&, T const&) const;

private:
    typedef std::map<std::string, Entry> table;
    table tbl;

    // verify class invariant
    void validate() const;

    // decode
    bool fromString(std::string const&);

  };  // ParameterSet

  inline bool
  operator==(ParameterSet const& a, ParameterSet const& b) {
    return a.toStringOfTracked() == b.toStringOfTracked();
  }

  inline bool
  operator!=(ParameterSet const& a, ParameterSet const& b) {
    return !(a == b);
  }

  // specializations
  // ----------------------------------------------------------------------
  // Bool, vBool
  
  template<>
  inline bool
  ParameterSet::getParameter<bool>(std::string const& name) const {
    return retrieve(name).getBool();
  }
  
  // ----------------------------------------------------------------------
  // Int32, vInt32
  
  template<>
  inline int
  ParameterSet::getParameter<int>(std::string const& name) const {
    return retrieve(name).getInt32();
  }
  
  template<>
  inline std::vector<int>
  ParameterSet::getParameter<std::vector<int> >(std::string const& name) const {
    return retrieve(name).getVInt32();
  }
  
  // ----------------------------------------------------------------------
  // Uint32, vUint32
  
  template<>
  inline unsigned int
  ParameterSet::getParameter<unsigned int>(std::string const& name) const {
    return retrieve(name).getUInt32();
  }
  
  template<>
  inline std::vector<unsigned int>
  ParameterSet::getParameter<std::vector<unsigned int> >(std::string const& name) const {
    return retrieve(name).getVUInt32();
  }
  
  // ----------------------------------------------------------------------
  // Double, vDouble
  
  template<>
  inline double
  ParameterSet::getParameter<double>(std::string const& name) const {
    return retrieve(name).getDouble();
  }
  
  template<>
  inline std::vector<double>
  ParameterSet::getParameter<std::vector<double> >(std::string const& name) const {
    return retrieve(name).getVDouble();
  }
  
  // ----------------------------------------------------------------------
  // String, vString
  
  template<>
  inline std::string
  ParameterSet::getParameter<std::string>(std::string const& name) const {
    return retrieve(name).getString();
  }
  
  template<>
  inline std::vector<std::string>
  ParameterSet::getParameter<std::vector<std::string> >(std::string const& name) const {
    return retrieve(name).getVString();
  }
  
  // ----------------------------------------------------------------------
  // PSet, vPSet
  
  template<>
  inline ParameterSet::ParameterSet
  ParameterSet::getParameter<edm::ParameterSet>(std::string const& name) const {
    return retrieve(name).getPSet();
  }
  
  template<>
  inline std::vector<ParameterSet::ParameterSet>
  ParameterSet::getParameter<std::vector<edm::ParameterSet> >(std::string const& name) const {
    return retrieve(name).getVPSet();
  }
  
  // untracked parameters
  
  // ----------------------------------------------------------------------
  // Bool, vBool
  
  template<>
  inline bool
  ParameterSet::getUntrackedParameter<bool>(std::string const& name, bool const& default_) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? default_ : entryPtr->getBool();
  }
  
  // ----------------------------------------------------------------------
  // Int32, vInt32
  
  template<>
  inline int
  ParameterSet::getUntrackedParameter<int>(std::string const& name, int const& default_) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? default_ : entryPtr->getInt32();
  }
  
  template<>
  inline std::vector<int>
  ParameterSet::getUntrackedParameter<std::vector<int> >(std::string const& name, std::vector<int> const& default_) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? default_ : entryPtr->getVInt32();
  }
  
  // ----------------------------------------------------------------------
  // Uint32, vUint32
  
  template<>
  inline unsigned int
  ParameterSet::getUntrackedParameter<unsigned int>(std::string const& name, unsigned int const& default_) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? default_ : entryPtr->getUInt32();
  }
  
  template<>
  inline std::vector<unsigned int>
  ParameterSet::getUntrackedParameter<std::vector<unsigned int> >(std::string const& name, std::vector<unsigned int> const& default_) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? default_ : entryPtr->getVUInt32();
  }
  
  // ----------------------------------------------------------------------
  // Double, vDouble
  
  template<>
  inline double
  ParameterSet::getUntrackedParameter<double>(std::string const& name, double const& default_) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? default_ : entryPtr->getDouble();
  }
  
  template<>
  inline std::vector<double>
  ParameterSet::getUntrackedParameter<std::vector<double> >(std::string const& name, std::vector<double> const& default_) const {
    Entry const* entryPtr = retrieveUntracked(name); return entryPtr == 0 ? default_ : entryPtr->getVDouble(); }
  
  // ----------------------------------------------------------------------
  // String, vString
  
  template<>
  inline std::string
  ParameterSet::getUntrackedParameter<std::string>(std::string const& name, std::string const& default_) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? default_ : entryPtr->getString();
  }
  
  template<>
  inline std::vector<std::string>
  ParameterSet::getUntrackedParameter<std::vector<std::string> >(std::string const& name, std::vector<std::string> const& default_) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? default_ : entryPtr->getVString();
  }
  
  // ----------------------------------------------------------------------
  // PSet, vPSet
  
  template<>
  inline ParameterSet::ParameterSet
  ParameterSet::getUntrackedParameter<edm::ParameterSet>(std::string const& name, edm::ParameterSet const& default_) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? default_ : entryPtr->getPSet();
  }
  
  template<>
  inline std::vector<ParameterSet::ParameterSet>
  ParameterSet::getUntrackedParameter<std::vector<edm::ParameterSet> >(std::string const& name, std::vector<edm::ParameterSet> const& default_) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? default_ : entryPtr->getVPSet();
  }
  
}  // namespace edm

// epilog

#endif  // PARAMETERSET_H

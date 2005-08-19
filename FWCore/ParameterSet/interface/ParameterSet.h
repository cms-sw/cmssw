// ----------------------------------------------------------------------
// $Id: ParameterSet.h,v 1.6 2005/06/23 19:57:22 wmtan Exp $
//
// Declaration for ParameterSet(parameter set) and related types
// ----------------------------------------------------------------------


// ----------------------------------------------------------------------
// prolog

#ifndef  PARAMETERSET_H
#define  PARAMETERSET_H


// ----------------------------------------------------------------------
// prerequisite source files and headers

#include "FWCore/EDProduct/interface/ParameterSetID.h"
#include "FWCore/ParameterSet/interface/Entry.h"
#include <string>
#include <map>
#include <stdexcept>
#include <vector>


// ----------------------------------------------------------------------
// contents

namespace edm {

  class ParameterSet {
  public:
    // default-construct
    ParameterSet();

    // construct from coded string
    explicit ParameterSet(std::string const&);

    // identification
    ParameterSetID id() const;

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
    void 
    addParameter(std::string const& name, T value)
    {
      invalidate();
      insert(true, name, Entry(value, true));
    }

    template< class T >
    T
    getUntrackedParameter(std::string const&, T const&) const;

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

    // decode
    bool fromString(std::string const&);

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
  // PSet, vPSet
  
  template<>
  inline 
  ParameterSet::ParameterSet
  ParameterSet::getParameter<edm::ParameterSet>(std::string const& name) const {
    return retrieve(name).getPSet();
  }
  
  template<>
  inline 
  std::vector<ParameterSet::ParameterSet>
  ParameterSet::getParameter<std::vector<edm::ParameterSet> >(std::string const& name) const {
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
  std::vector<int>
  ParameterSet::getUntrackedParameter<std::vector<int> >(std::string const& name, std::vector<int> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getVInt32();
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
  std::vector<unsigned int>
  ParameterSet::getUntrackedParameter<std::vector<unsigned int> >(std::string const& name, std::vector<unsigned int> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getVUInt32();
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
  std::vector<double>
  ParameterSet::getUntrackedParameter<std::vector<double> >(std::string const& name, std::vector<double> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name); return entryPtr == 0 ? defaultValue : entryPtr->getVDouble(); }
  
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
  std::vector<std::string>
  ParameterSet::getUntrackedParameter<std::vector<std::string> >(std::string const& name, std::vector<std::string> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getVString();
  }
  
  // ----------------------------------------------------------------------
  // PSet, vPSet
  
  template<>
  inline 
  ParameterSet
  ParameterSet::getUntrackedParameter<edm::ParameterSet>(std::string const& name, edm::ParameterSet const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getPSet();
  }
  
  template<>
  inline 
  std::vector<ParameterSet::ParameterSet>
  ParameterSet::getUntrackedParameter<std::vector<edm::ParameterSet> >(std::string const& name, std::vector<edm::ParameterSet> const& defaultValue) const {
    Entry const* entryPtr = retrieveUntracked(name);
    return entryPtr == 0 ? defaultValue : entryPtr->getVPSet();
  }
  
}  // namespace edm

// epilog

#endif  // PARAMETERSET_H

// ----------------------------------------------------------------------
// $Id: ParameterSet.h,v 1.3 2005/06/13 23:59:09 wmtan Exp $
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
    explicit ParameterSetError( std::string const& mesg ) : std::runtime_error( mesg ) {}

    virtual ~ParameterSetError() throw() {}

  };  // ParameterSetError


// ----------------------------------------------------------------------
// edm::ParameterSet

  class ParameterSet {
  public:
    // default-construct
    ParameterSet() : tbl() {}

    // construct from coded string
    explicit ParameterSet( std::string const& );

    // Entry-handling
    Entry const& retrieve( std::string const& ) const;
    Entry const* const retrieveUntracked( std::string const& ) const;
    void insert( bool ok_to_replace, std::string const& , Entry const& );
    void augment( ParameterSet const& from );

    // encode
    std::string toString() const;
    std::string toStringOfTracked() const;

private:
    typedef std::map<std::string, Entry> table;
    table tbl;

    // verify class invariant
    void validate() const;

    // decode
    bool fromString( std::string const& );

  };  // ParameterSet

  inline bool
  operator==( ParameterSet const& a, ParameterSet const& b ) {
    return a.toStringOfTracked() == b.toStringOfTracked();
  }

  inline bool
  operator!=( ParameterSet const& a, ParameterSet const& b ) {
    return !(a == b);
  }

  template< class T >
  T
  getParameter( ParameterSet const&, std::string const& );

  template< class T >
  T
  getUntrackedParameter( ParameterSet const&, std::string const&, T const&);

  // specializations
  template<>
  bool
  getParameter<bool>( ParameterSet const&, std::string const& );
  template<>
  std::vector<bool>
  getParameter<std::vector<bool> >( ParameterSet const&, std::string const& );
  template<>
  int
  getParameter<int>( ParameterSet const&, std::string const& );
  template<>
  std::vector<int>
  getParameter<std::vector<int> >( ParameterSet const&, std::string const& );
  template<>
  unsigned int
  getParameter<unsigned int>( ParameterSet const&, std::string const& );
  template<>
  std::vector<unsigned int>
  getParameter<std::vector<unsigned int> >( ParameterSet const&, std::string const& );
  template<>
  double
  getParameter<double>( ParameterSet const&, std::string const& );
  template<>
  std::vector<double>
  getParameter<std::vector<double> >( ParameterSet const&, std::string const& );
  template<>
  std::string
  getParameter<std::string>( ParameterSet const&, std::string const& );
  template<>
  std::vector<std::string>
  getParameter<std::vector<std::string> >( ParameterSet const&, std::string const& );
  template<>
  ParameterSet
  getParameter<ParameterSet>( ParameterSet const&, std::string const& );
  template<>
  std::vector<ParameterSet>
  getParameter<std::vector<ParameterSet> >( ParameterSet const&, std::string const& );

  template<>
  bool
  getUntrackedParameter<bool>( ParameterSet const&, std::string const&, bool const& );
  template<>
  std::vector<bool>
  getUntrackedParameter<std::vector<bool> >( ParameterSet const&, std::string const&, std::vector<bool> const& );
  template<>
  int
  getUntrackedParameter<int>( ParameterSet const&, std::string const&, int const& );
  template<>
  std::vector<int>
  getUntrackedParameter<std::vector<int> >( ParameterSet const&, std::string const&, std::vector<int> const& );
  template<>
  unsigned int
  getUntrackedParameter<unsigned int>( ParameterSet const&, std::string const&, unsigned int const& );
  template<>
  std::vector<unsigned int>
  getUntrackedParameter<std::vector<unsigned int> >( ParameterSet const&, std::string const&, std::vector<unsigned int> const& );
  template<>
  double
  getUntrackedParameter<double>( ParameterSet const&, std::string const&, double const& );
  template<>
  std::vector<double>
  getUntrackedParameter<std::vector<double> >( ParameterSet const&, std::string const&, std::vector<double> const& );
  template<>
  std::string
  getUntrackedParameter<std::string>( ParameterSet const&, std::string const&, std::string const& );
  template<>
  std::vector<std::string>
  getUntrackedParameter<std::vector<std::string> >( ParameterSet const&, std::string const&, std::vector<std::string> const& );
  template<>
  ParameterSet
  getUntrackedParameter<ParameterSet>( ParameterSet const&, std::string const&, ParameterSet const& );
  template<>
  std::vector<ParameterSet>
  getUntrackedParameter<std::vector<ParameterSet> >( ParameterSet const&, std::string const&, std::vector<ParameterSet> const& );
}  // namespace edm

// ----------------------------------------------------------------------
// Bool, vBool

template<>
inline bool
edm::getParameter<bool>( ParameterSet const& p, std::string const& name ) {
  return p.retrieve(name).getBool();
}

// ----------------------------------------------------------------------
// Int32, vInt32

template<>
inline int
edm::getParameter<int>( ParameterSet const& p, std::string const& name ) {
  return p.retrieve(name).getInt32();
}

template<>
inline std::vector<int>
edm::getParameter<std::vector<int> >( ParameterSet const& p, std::string const& name ) {
  return p.retrieve(name).getVInt32();
}

// ----------------------------------------------------------------------
// Uint32, vUint32

template<>
inline unsigned int
edm::getParameter<unsigned int>( ParameterSet const& p, std::string const& name ) {
  return p.retrieve(name).getUInt32();
}

template<>
inline std::vector<unsigned int>
edm::getParameter<std::vector<unsigned int> >( ParameterSet const& p, std::string const& name ) {
  return p.retrieve(name).getVUInt32();
}

// ----------------------------------------------------------------------
// Double, vDouble

template<>
inline double
edm::getParameter<double>( ParameterSet const& p, std::string const& name ) {
  return p.retrieve(name).getDouble();
}

template<>
inline std::vector<double>
edm::getParameter<std::vector<double> >( ParameterSet const& p, std::string const& name ) {
  return p.retrieve(name).getVDouble();
}

// ----------------------------------------------------------------------
// String, vString

template<>
inline std::string
edm::getParameter<std::string>( ParameterSet const& p, std::string const& name ) {
  return p.retrieve(name).getString();
}

template<>
inline std::vector<std::string>
edm::getParameter<std::vector<std::string> >( ParameterSet const& p, std::string const& name ) {
  return p.retrieve(name).getVString();
}

// ----------------------------------------------------------------------
// PSet, vPSet

template<>
inline edm::ParameterSet
edm::getParameter<edm::ParameterSet>( ParameterSet const& p, std::string const& name ) {
  return p.retrieve(name).getPSet();
}

template<>
inline std::vector<edm::ParameterSet>
edm::getParameter<std::vector<edm::ParameterSet> >( ParameterSet const& p, std::string const& name ) {
  return p.retrieve(name).getVPSet();
}

// untracked parameters

// ----------------------------------------------------------------------
// Bool, vBool

template<>
inline bool
edm::getUntrackedParameter<bool>( ParameterSet const& p, std::string const& name, bool const& default_ ) {
  Entry const* entryPtr = p.retrieveUntracked(name);
  return entryPtr == 0 ? default_ : entryPtr->getBool();
}

// ----------------------------------------------------------------------
// Int32, vInt32

template<>
inline int
edm::getUntrackedParameter<int>( ParameterSet const& p, std::string const& name, int const& default_ ) {
  Entry const* entryPtr = p.retrieveUntracked(name);
  return entryPtr == 0 ? default_ : entryPtr->getInt32();
}

template<>
inline std::vector<int>
edm::getUntrackedParameter<std::vector<int> >( ParameterSet const& p, std::string const& name, std::vector<int> const& default_ ) {
  Entry const* entryPtr = p.retrieveUntracked(name);
  return entryPtr == 0 ? default_ : entryPtr->getVInt32();
}

// ----------------------------------------------------------------------
// Uint32, vUint32

template<>
inline unsigned int
edm::getUntrackedParameter<unsigned int>( ParameterSet const& p, std::string const& name, unsigned int const& default_ ) {
  Entry const* entryPtr = p.retrieveUntracked(name);
  return entryPtr == 0 ? default_ : entryPtr->getUInt32();
}

template<>
inline std::vector<unsigned int>
edm::getUntrackedParameter<std::vector<unsigned int> >( ParameterSet const& p, std::string const& name, std::vector<unsigned int> const& default_ ) {
  Entry const* entryPtr = p.retrieveUntracked(name);
  return entryPtr == 0 ? default_ : entryPtr->getVUInt32();
}

// ----------------------------------------------------------------------
// Double, vDouble

template<>
inline double
edm::getUntrackedParameter<double>( ParameterSet const& p, std::string const& name, double const& default_ ) {
  Entry const* entryPtr = p.retrieveUntracked(name);
  return entryPtr == 0 ? default_ : entryPtr->getDouble();
}

template<>
inline std::vector<double>
edm::getUntrackedParameter<std::vector<double> >( ParameterSet const& p, std::string const& name, std::vector<double> const& default_ ) {
  Entry const* entryPtr = p.retrieveUntracked(name); return entryPtr == 0 ? default_ : entryPtr->getVDouble(); }

// ----------------------------------------------------------------------
// String, vString

template<>
inline std::string
edm::getUntrackedParameter<std::string>( ParameterSet const& p, std::string const& name, std::string const& default_ ) {
  Entry const* entryPtr = p.retrieveUntracked(name);
  return entryPtr == 0 ? default_ : entryPtr->getString();
}

template<>
inline std::vector<std::string>
edm::getUntrackedParameter<std::vector<std::string> >( ParameterSet const& p, std::string const& name, std::vector<std::string> const& default_ ) {
  Entry const* entryPtr = p.retrieveUntracked(name);
  return entryPtr == 0 ? default_ : entryPtr->getVString();
}

// ----------------------------------------------------------------------
// PSet, vPSet

template<>
inline edm::ParameterSet
edm::getUntrackedParameter<edm::ParameterSet>( ParameterSet const& p, std::string const& name, edm::ParameterSet const& default_ ) {
  Entry const* entryPtr = p.retrieveUntracked(name);
  return entryPtr == 0 ? default_ : entryPtr->getPSet();
}

template<>
inline std::vector<edm::ParameterSet>
edm::getUntrackedParameter<std::vector<edm::ParameterSet> >( ParameterSet const& p, std::string const& name, std::vector<edm::ParameterSet> const& default_ ) {
  Entry const* entryPtr = p.retrieveUntracked(name);
  return entryPtr == 0 ? default_ : entryPtr->getVPSet();
}

// epilog

#endif  // PARAMETERSET_H

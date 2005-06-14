// ----------------------------------------------------------------------
// $Id: ParameterSet.h,v 1.2 2005/06/10 03:57:11 wmtan Exp $
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
  getP( ParameterSet const&, std::string const& );

  template< class T >
  T
  getUntrackedP( ParameterSet const&, std::string const&, T const&);

  // specializations
  template<>
  bool
  getP<bool>( ParameterSet const&, std::string const& );
  template<>
  std::vector<bool>
  getP<std::vector<bool> >( ParameterSet const&, std::string const& );
  template<>
  int
  getP<int>( ParameterSet const&, std::string const& );
  template<>
  std::vector<int>
  getP<std::vector<int> >( ParameterSet const&, std::string const& );
  template<>
  unsigned int
  getP<unsigned int>( ParameterSet const&, std::string const& );
  template<>
  std::vector<unsigned int>
  getP<std::vector<unsigned int> >( ParameterSet const&, std::string const& );
  template<>
  double
  getP<double>( ParameterSet const&, std::string const& );
  template<>
  std::vector<double>
  getP<std::vector<double> >( ParameterSet const&, std::string const& );
  template<>
  std::string
  getP<std::string>( ParameterSet const&, std::string const& );
  template<>
  std::vector<std::string>
  getP<std::vector<std::string> >( ParameterSet const&, std::string const& );
  template<>
  ParameterSet
  getP<ParameterSet>( ParameterSet const&, std::string const& );
  template<>
  std::vector<ParameterSet>
  getP<std::vector<ParameterSet> >( ParameterSet const&, std::string const& );

  template<>
  bool
  getUntrackedP<bool>( ParameterSet const&, std::string const&, bool const& );
  template<>
  std::vector<bool>
  getUntrackedP<std::vector<bool> >( ParameterSet const&, std::string const&, std::vector<bool> const& );
  template<>
  int
  getUntrackedP<int>( ParameterSet const&, std::string const&, int const& );
  template<>
  std::vector<int>
  getUntrackedP<std::vector<int> >( ParameterSet const&, std::string const&, std::vector<int> const& );
  template<>
  unsigned int
  getUntrackedP<unsigned int>( ParameterSet const&, std::string const&, unsigned int const& );
  template<>
  std::vector<unsigned int>
  getUntrackedP<std::vector<unsigned int> >( ParameterSet const&, std::string const&, std::vector<unsigned int> const& );
  template<>
  double
  getUntrackedP<double>( ParameterSet const&, std::string const&, double const& );
  template<>
  std::vector<double>
  getUntrackedP<std::vector<double> >( ParameterSet const&, std::string const&, std::vector<double> const& );
  template<>
  std::string
  getUntrackedP<std::string>( ParameterSet const&, std::string const&, std::string const& );
  template<>
  std::vector<std::string>
  getUntrackedP<std::vector<std::string> >( ParameterSet const&, std::string const&, std::vector<std::string> const& );
}  // namespace edm

// ----------------------------------------------------------------------
// Bool, vBool

template<>
inline bool
edm::getP<bool>( ParameterSet const& p, std::string const& name ) {
  return p.retrieve(name).getBool();
}

// ----------------------------------------------------------------------
// Int32, vInt32

template<>
inline int
edm::getP<int>( ParameterSet const& p, std::string const& name ) {
  return p.retrieve(name).getInt32();
}

template<>
inline std::vector<int>
edm::getP<std::vector<int> >( ParameterSet const& p, std::string const& name ) {
  return p.retrieve(name).getVInt32();
}

// ----------------------------------------------------------------------
// Uint32, vUint32

template<>
inline unsigned int
edm::getP<unsigned int>( ParameterSet const& p, std::string const& name ) {
  return p.retrieve(name).getUInt32();
}

template<>
inline std::vector<unsigned int>
edm::getP<std::vector<unsigned int> >( ParameterSet const& p, std::string const& name ) {
  return p.retrieve(name).getVUInt32();
}

// ----------------------------------------------------------------------
// Double, vDouble

template<>
inline double
edm::getP<double>( ParameterSet const& p, std::string const& name ) {
  return p.retrieve(name).getDouble();
}

template<>
inline std::vector<double>
edm::getP<std::vector<double> >( ParameterSet const& p, std::string const& name ) {
  return p.retrieve(name).getVDouble();
}

// ----------------------------------------------------------------------
// String, vString

template<>
inline std::string
edm::getP<std::string>( ParameterSet const& p, std::string const& name ) {
  return p.retrieve(name).getString();
}

template<>
inline std::vector<std::string>
edm::getP<std::vector<std::string> >( ParameterSet const& p, std::string const& name ) {
  return p.retrieve(name).getVString();
}

// ----------------------------------------------------------------------
// PSet, vPSet

template<>
inline edm::ParameterSet
edm::getP<edm::ParameterSet>( ParameterSet const& p, std::string const& name ) {
  return p.retrieve(name).getPSet();
}

template<>
inline std::vector<edm::ParameterSet>
edm::getP<std::vector<edm::ParameterSet> >( ParameterSet const& p, std::string const& name ) {
  return p.retrieve(name).getVPSet();
}

// untracked parameters

// ----------------------------------------------------------------------
// Bool, vBool

template<>
inline bool
edm::getUntrackedP<bool>( ParameterSet const& p, std::string const& name, bool const& default_ ) {
  Entry const* entryPtr = p.retrieveUntracked(name);
  return entryPtr == 0 ? default_ : entryPtr->getBool();
}

// ----------------------------------------------------------------------
// Int32, vInt32

template<>
inline int
edm::getUntrackedP<int>( ParameterSet const& p, std::string const& name, int const& default_ ) {
  Entry const* entryPtr = p.retrieveUntracked(name);
  return entryPtr == 0 ? default_ : entryPtr->getInt32();
}

template<>
inline std::vector<int>
edm::getUntrackedP<std::vector<int> >( ParameterSet const& p, std::string const& name, std::vector<int> const& default_ ) {
  Entry const* entryPtr = p.retrieveUntracked(name);
  return entryPtr == 0 ? default_ : entryPtr->getVInt32();
}

// ----------------------------------------------------------------------
// Uint32, vUint32

template<>
inline unsigned int
edm::getUntrackedP<unsigned int>( ParameterSet const& p, std::string const& name, unsigned int const& default_ ) {
  Entry const* entryPtr = p.retrieveUntracked(name);
  return entryPtr == 0 ? default_ : entryPtr->getUInt32();
}

template<>
inline std::vector<unsigned int>
edm::getUntrackedP<std::vector<unsigned int> >( ParameterSet const& p, std::string const& name, std::vector<unsigned int> const& default_ ) {
  Entry const* entryPtr = p.retrieveUntracked(name);
  return entryPtr == 0 ? default_ : entryPtr->getVUInt32();
}

// ----------------------------------------------------------------------
// Double, vDouble

template<>
inline double
edm::getUntrackedP<double>( ParameterSet const& p, std::string const& name, double const& default_ ) {
  Entry const* entryPtr = p.retrieveUntracked(name);
  return entryPtr == 0 ? default_ : entryPtr->getDouble();
}

template<>
inline std::vector<double>
edm::getUntrackedP<std::vector<double> >( ParameterSet const& p, std::string const& name, std::vector<double> const& default_ ) {
  Entry const* entryPtr = p.retrieveUntracked(name);
  return entryPtr == 0 ? default_ : entryPtr->getVDouble();
}

// ----------------------------------------------------------------------
// String, vString

template<>
inline std::string
edm::getUntrackedP<std::string>( ParameterSet const& p, std::string const& name, std::string const& default_ ) {
  Entry const* entryPtr = p.retrieveUntracked(name);
  return entryPtr == 0 ? default_ : entryPtr->getString();
}

template<>
inline std::vector<std::string>
edm::getUntrackedP<std::vector<std::string> >( ParameterSet const& p, std::string const& name, std::vector<std::string> const& default_ ) {
  Entry const* entryPtr = p.retrieveUntracked(name);
  return entryPtr == 0 ? default_ : entryPtr->getVString();
}

// epilog

#endif  // PARAMETERSET_H

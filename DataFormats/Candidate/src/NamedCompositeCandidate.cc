// $Id: NamedCompositeCandidate.cc,v 1.4 2008/07/22 06:07:44 llista Exp $
#include "DataFormats/Candidate/interface/NamedCompositeCandidate.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <iostream> 

using namespace reco;

NamedCompositeCandidate::NamedCompositeCandidate(std::string name, 
						 const NamedCompositeCandidate::role_collection & roles,
						 const Candidate & c) :
  CompositeCandidate(c),
  name_(name),
  roles_(roles)
{

  // Check if there are the same number of daughters and roles
  int N1 = roles_.size();
  int N2 = numberOfDaughters();

  if ( N1 != N2 ) {
    throw cms::Exception("InvalidReference")
      << "NamedCompositeCandidate constructor: Number of roles and daughters differ, this is an error. Name = " << name << "\n";
  }
}

NamedCompositeCandidate::~NamedCompositeCandidate() { clearDaughters(); clearRoles(); }

NamedCompositeCandidate * NamedCompositeCandidate::clone() const { return new NamedCompositeCandidate( * this ); }

void NamedCompositeCandidate::applyRoles()
{

  // Check if there are the same number of daughters and roles
  int N1 = roles_.size();
  int N2 = numberOfDaughters();
  if ( N1 != N2 ) {
    throw cms::Exception("InvalidReference")
      << "NamedCompositeCandidate::applyRoles : Number of roles and daughters differ, this is an error.\n";
  }
  // Set up the daughter roles
  for ( int i = 0 ; i < N1; ++i ) {
    std::string role = roles_[i];
    Candidate * c = CompositeCandidate::daughter( i );

    NamedCompositeCandidate * c1 = dynamic_cast<NamedCompositeCandidate *>(c);
    if ( c1 != 0 ) {
      c1->setName( role );
    }
  }
}

Candidate * NamedCompositeCandidate::daughter(const std::string& s ) 
{
  int ret = -1;
  int i = 0, N = roles_.size();
  bool found = false;
  for ( ; i < N && !found; ++i ) {
    if ( s == roles_[i] ) {
      found = true;
      ret = i;
    }
  }

  if ( ret < 0 ) {
    throw cms::Exception("InvalidReference")
      << "NamedCompositeCandidate::daughter: Cannot find role " << s << "\n";
  }
  
  return daughter(ret);
}

const Candidate * NamedCompositeCandidate::daughter(const std::string& s ) const 
{
  int ret = -1;
  int i = 0, N = roles_.size();
  bool found = false;
  for ( ; i < N && !found; ++i ) {
    if ( s == roles_[i] ) {
      found = true;
      ret = i;
    }
  }

  if ( ret < 0 ) {
    throw cms::Exception("InvalidReference")
      << "NamedCompositeCandidate::daughter: Cannot find role " << s << "\n";
  }
  
  return daughter(ret);
}

void NamedCompositeCandidate::addDaughter( const Candidate & cand, const std::string& s )
{

  role_collection::iterator begin = roles_.begin(), end = roles_.end();
  bool isFound = ( find( begin, end, s) != end );
  if ( isFound ) {
    throw cms::Exception("InvalidReference")
      << "NamedCompositeCandidate::addDaughter: Already have role with name " << s 
      << ", please clearDaughters, or use a new name\n";
  }

  roles_.push_back( s );
  std::auto_ptr<Candidate> c( cand.clone() );
  NamedCompositeCandidate * c1 =  dynamic_cast<NamedCompositeCandidate*>(&*c);
  if ( c1 != 0 ) {
    c1->setName( s );
  }
  CompositeCandidate::addDaughter( c );
}

void NamedCompositeCandidate::addDaughter( std::auto_ptr<Candidate> cand, const std::string& s )
{
  role_collection::iterator begin = roles_.begin(), end = roles_.end();
  bool isFound = ( find( begin, end, s) != end );
  if ( isFound ) {
    throw cms::Exception("InvalidReference")
      << "NamedCompositeCandidate::addDaughter: Already have role with name " << s 
      << ", please clearDaughters, or use a new name\n";
  }

  roles_.push_back( s );
  NamedCompositeCandidate * c1 = dynamic_cast<NamedCompositeCandidate*>(&*cand);
  if ( c1 != 0 ) {
    c1->setName( s );
  }
  CompositeCandidate::addDaughter( cand );
}



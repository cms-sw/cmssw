// $Id: NamedCompositeCandidate.cc,v 1.5.4.1 2007/11/30 13:16:01 llista Exp $
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
  size_t n = c.numberOfDaughters();
  if ( roles_.size() == n ) {
    for(size_t i = 0; i != n; ++i)
      addDaughter(*c.daughter(i), roles[i] );
  } else {
    for(size_t i = 0; i != n; ++i)
      addDaughter(*c.daughter(i), "" );
  }
}

NamedCompositeCandidate::~NamedCompositeCandidate() { }

NamedCompositeCandidate * NamedCompositeCandidate::clone() const { return new NamedCompositeCandidate( * this ); }

void NamedCompositeCandidate::applyRoles()
{

  int N1 = roles_.size();
  int N2 = numberOfDaughters();

  if ( N1 != N2 ) {
    std::cout << "NamedCompositeCandidate: Number of roles and role candidates differ, exiting" << std::endl;
    return;
  }

  for ( int i = 0 ; i < N1; ++i ) {
    std::string role = roles_[i];
    Candidate * c = CompositeCandidate::daughter( i );

    NamedCompositeCandidate * c1 = dynamic_cast<NamedCompositeCandidate *>(c);
    if ( c1 != 0 ) {
      c1->setName( role );
    }
  }
}

Candidate * NamedCompositeCandidate::daughter( std::string s ) 
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
    std::cout << "NamedCompositeCandidate::daughter: Cannot find role " << s << std::endl;
    return 0;
  }
  
  return daughter(ret);
}

const Candidate * NamedCompositeCandidate::daughter( std::string s ) const 
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
    std::cout << "NamedCompositeCandidate::daughter: Cannot find role " << s << std::endl;
    return 0;
  }
  
  return daughter(ret);
}

void NamedCompositeCandidate::addDaughter( const Candidate & cand, std::string s )
{

  role_collection::iterator begin = roles_.begin(), end = roles_.end();
  bool isFound = ( find( begin, end, s) != end );
  if ( isFound ) {
    std::cout << "NamedCompositeCandidate::addDaughter: Already have role with name " << s 
	      << ", please clearDaughters, or use a new name" << std::endl;
    return;
  }

  roles_.push_back( s );
  std::auto_ptr<Candidate> c( cand.clone() );
  NamedCompositeCandidate * c1 =  dynamic_cast<NamedCompositeCandidate*>(&*c);
  if ( c1 != 0 ) {
    c1->setName( s );
  }
  CompositeCandidate::addDaughter( c );
}

void NamedCompositeCandidate::addDaughter( std::auto_ptr<Candidate> cand, std::string s )
{
  role_collection::iterator begin = roles_.begin(), end = roles_.end();
  bool isFound = ( find( begin, end, s) != end );
  if ( isFound ) {
    std::cout << "NamedCompositeCandidate::addDaughter: Already have role with name " << s 
	      << ", please clearDaughters, or use a new name" << std::endl;
    return;
  }

  roles_.push_back( s );
  NamedCompositeCandidate * c1 = dynamic_cast<NamedCompositeCandidate*>(&*cand);
  if ( c1 != 0 ) {
    c1->setName( s );
  }
  CompositeCandidate::addDaughter( cand );
}

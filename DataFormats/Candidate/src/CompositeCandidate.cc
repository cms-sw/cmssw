// $Id: CompositeCandidate.cc,v 1.8 2007/11/30 14:03:28 llista Exp $
#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace reco;

CompositeCandidate::CompositeCandidate(const Candidate & c,
				       std::string name ) :
  Candidate(c), name_(name) {
  size_t n = c.numberOfDaughters();
  for(size_t i = 0; i != n; ++i) {
      addDaughter(*c.daughter(i));
  }
}

CompositeCandidate::CompositeCandidate(const Candidate & c,
				       std::string name,
				       role_collection const & roles) :
  Candidate(c), name_(name), roles_(roles) {
  size_t n = c.numberOfDaughters();
  size_t r = roles_.size();
  bool sameSize = ( n == r );
  for(size_t i = 0; i != n; ++i) {
    if ( sameSize ) 
      addDaughter(*c.daughter(i), roles_[i]);
    else
      addDaughter(*c.daughter(i));
  }
}

CompositeCandidate::~CompositeCandidate() { }

CompositeCandidate * CompositeCandidate::clone() const { return new CompositeCandidate( * this ); }

Candidate::const_iterator CompositeCandidate::begin() const { return const_iterator( new const_iterator_imp_specific( dau.begin() ) ); }

Candidate::const_iterator CompositeCandidate::end() const { return const_iterator( new const_iterator_imp_specific( dau.end() ) ); }    

Candidate::iterator CompositeCandidate::begin() { return iterator( new iterator_imp_specific( dau.begin() ) ); }

Candidate::iterator CompositeCandidate::end() { return iterator( new iterator_imp_specific( dau.end() ) ); }    

const Candidate * CompositeCandidate::daughter( size_type i ) const { 
  return ( i >= 0 && i < numberOfDaughters() ) ? & dau[ i ] : 0;
}

Candidate * CompositeCandidate::daughter( size_type i ) { 
  Candidate * d = ( i >= 0 && i < numberOfDaughters() ) ? & dau[ i ] : 0;
  return d;
}

const Candidate * CompositeCandidate::mother( size_type i ) const { 
  return 0;
}

size_t CompositeCandidate::numberOfDaughters() const { return dau.size(); }

size_t CompositeCandidate::numberOfMothers() const { return 0; }

bool CompositeCandidate::overlap( const Candidate & c2 ) const {
  throw cms::Exception( "Error" ) << "can't check overlap internally for CompositeCanddate";
}


void CompositeCandidate::applyRoles()
{

  // Check if there are the same number of daughters and roles
  int N1 = roles_.size();
  int N2 = numberOfDaughters();
  if ( N1 != N2 ) {
    throw cms::Exception("InvalidReference")
      << "CompositeCandidate::applyRoles : Number of roles and daughters differ, this is an error.\n";
  }
  // Set up the daughter roles
  for ( int i = 0 ; i < N1; ++i ) {
    std::string role = roles_[i];
    Candidate * c = CompositeCandidate::daughter( i );

    CompositeCandidate * c1 = dynamic_cast<CompositeCandidate *>(c);
    if ( c1 != 0 ) {
      c1->setName( role );
    }
  }
}

Candidate * CompositeCandidate::daughter( std::string s ) 
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
      << "CompositeCandidate::daughter: Cannot find role " << s << "\n";
  }
  
  return daughter(ret);
}

const Candidate * CompositeCandidate::daughter( std::string s ) const 
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
      << "CompositeCandidate::daughter: Cannot find role " << s << "\n";
  }
  
  return daughter(ret);
}

void CompositeCandidate::addDaughter( const Candidate & cand, std::string s )
{

  role_collection::iterator begin = roles_.begin(), end = roles_.end();
  bool isFound = ( find( begin, end, s) != end );
  if ( isFound ) {
    throw cms::Exception("InvalidReference")
      << "CompositeCandidate::addDaughter: Already have role with name " << s 
      << ", please clearDaughters, or use a new name\n";
  }

  roles_.push_back( s );
  std::auto_ptr<Candidate> c( cand.clone() );
  CompositeCandidate * c1 =  dynamic_cast<CompositeCandidate*>(&*c);
  if ( c1 != 0 ) {
    c1->setName( s );
  }
  CompositeCandidate::addDaughter( c );
}

void CompositeCandidate::addDaughter( std::auto_ptr<Candidate> cand, std::string s )
{
  role_collection::iterator begin = roles_.begin(), end = roles_.end();
  bool isFound = ( find( begin, end, s) != end );
  if ( isFound ) {
    throw cms::Exception("InvalidReference")
      << "CompositeCandidate::addDaughter: Already have role with name " << s 
      << ", please clearDaughters, or use a new name\n";
  }

  roles_.push_back( s );
  CompositeCandidate * c1 = dynamic_cast<CompositeCandidate*>(&*cand);
  if ( c1 != 0 ) {
    c1->setName( s );
  }
  CompositeCandidate::addDaughter( cand );
}



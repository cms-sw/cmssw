#ifndef UtilAlgos_ObjectCounter_h
#define UtilAlgos_ObjectCounter_h
// Merges multiple collections
// $Id: ObjectCounter.h,v 1.1 2005/12/13 03:39:04 llista Exp $
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <iostream>
#include <cmath>

template<typename C>
class ObjectCounter : public edm::EDAnalyzer {
public:
  explicit ObjectCounter( const edm::ParameterSet& );
  ~ObjectCounter();

private:
  virtual void analyze( const edm::Event&, const edm::EventSetup& );
  std::string src_;
  bool verbose_;
  unsigned long n_, nSum_, n2Sum_;
};

template<typename C>
ObjectCounter<C>::ObjectCounter( const edm::ParameterSet& par ) : 
  src_( par.template getParameter<std::string>( "src" ) ), 
  verbose_( par.template getUntrackedParameter<bool>( "verbose", true ) ),
  n_( 0 ), nSum_( 0 ), n2Sum_( 0 ) {
}

template<typename C>
ObjectCounter<C>::~ObjectCounter() {
  double n = double( nSum_ ) / n_, n2 = double ( n2Sum_ ) / n_;
  double s = sqrt( n2 - n * n );
  if ( verbose_ ) 
    std::cout << ">>> Entries in collection " << src_ << ": " << n << " +/- " << s << std::endl;
}

template<typename C>
void ObjectCounter<C>::analyze( const edm::Event& evt, const edm::EventSetup& ) {
  std::auto_ptr<C> coll( new C );
  edm::Handle<C> h;
  evt.getByLabel( src_, h );
  int n = h->size();
  nSum_ += n;
  n2Sum_ += ( n * n );
  ++ n_;
}

#endif

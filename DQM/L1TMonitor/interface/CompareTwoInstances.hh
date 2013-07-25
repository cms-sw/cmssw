#ifndef COMPARETWOINSTANCES_HH
#define COMPARETWOINSTANCES_HH
// -*- C++ -*-
//
// 
/**\class CompareTwoInstances

 Description: Compare two things that you think are the same, plot in2d histo

 Implementation:
    + Use templates to do comparison, store in TH2D
    + Assumption: that data in the emulated and real banks is strictly 
      comparable, i.e., that it is even ordered the same in data and MC.
      If this is not the case we'll have to rethink.

*/
//
// Original Author:  Peter Wittich
//         Created:  Mon Jan  8 21:11:04 CET 2007
// $Id: CompareTwoInstances.hh,v 1.1 2007/02/20 22:49:54 wittich Exp $
// $Log: CompareTwoInstances.hh,v $
// Revision 1.1  2007/02/20 22:49:54  wittich
// - templated class to facilitate comparison between data and emulation.
//   see file for caveats.
//
//


// system include files
#include <iostream>

// FW include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

// Root
#include "TH2.h"

// User Include Files


// Templated helper class
template< class T, class Res>
class CompareTwoInstances {
private:
  // pointer to a member function of T which has no arguments 
  // and returns type Res.
  Res (T::* _s)() const;
  TH2D *_2dhist;
public:
  typedef std::vector<T> TCollection;


  // Constructor
  // pass in member funtion which returns data to be compared
  CompareTwoInstances( Res (T::*s)() const, const char *hname, 
		       const char *htitle, int nbins, float xlow, float xhi) :
    _s(s),
    _2dhist(new TH2D(hname, htitle, nbins, xlow, xhi, nbins, xlow, xhi))
  {}
 private:
  // default constructor - don't want this to be visible
  CompareTwoInstances() :
    _s(0),
    _2dhist(0)
  {}
 public:
  void 
  operator()( edm::Handle<TCollection> h1, 
	      edm::Handle<TCollection> h2)
  {
    if ( !(h1.isValid() && h2.isValid()) ) {
      // need better error mechanism
      std::cout << "Invalid handle(s) passed into operator()" << std::endl;
      return;
    }
    // need typename here, go figure.....
    typename TCollection::const_iterator i1, i2;
    for ( i1 = h1->begin(); i1 != h1->end(); ++ i1 ) {
      for ( i2 = h2->begin(); i2 != h2->end(); ++ i2 ) {
 	_2dhist->Fill(((*i1).*_s)(), ((*i2).*_s)());
      }
    }
  }
  const TH2D *data () const {
    return _2dhist;
  }
  // need to hook this into DQM 
};

// How to use this - example
//
// class declaration
//
// In your class define two collections and their types and input tags
//   edm::InputTag Coll1_, Coll2_;
//   CompareTwoInstances<L1GctEmCand, unsigned> a_;

// initialize in the constructor
//  Coll1_ = iConfig.getUntrackedParameter<edm::InputTag>("Coll1");
//  Coll2_ = iConfig.getUntrackedParameter<edm::InputTag>("Coll2");
/    
//  a_ = CompareTwoInstances<L1GctEmCand, unsigned>(
// 					  &L1GctEmCand::etaIndex,
// 					  "GctEmEta", "GCT EM ETA",
// 					  20, -10, 10);


// In the event entry, get the two collections and pass them to the 
// relevant guy.
//    using namespace edm;
//    Handle<L1GctEmCandCollection> p1, p2;
//    iEvent.getByLabel(Coll1_,p1);
//    iEvent.getByLabel(Coll2_,p2);
   
//    a_(p1,p2);
//
//    a_->data()->Draw(); // access to 2d histo.

#endif // COMPARETWOINSTANCES_HH

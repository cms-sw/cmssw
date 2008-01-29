#include "PhysicsTools/Utilities/src/CutSetter.h"
#include "PhysicsTools/Utilities/src/AndCombiner.h"
#include "PhysicsTools/Utilities/src/OrCombiner.h"
#include "PhysicsTools/Utilities/src/NotCombiner.h"
#ifdef BOOST_SPIRIT_DEBUG 
#include <iostream>
#endif
using namespace reco::parser;

void CutSetter::operator()( const char *, const char * ) const {
  if( 0 == cut_.get() ) {
    cut_ = selStack_.back();
    selStack_.pop_back();
  } else {
    switch ( cmbStack_.back() ) {
    case ( kAnd ) : {
      SelectorPtr lhs = cut_;
      cut_ = SelectorPtr( new AndCombiner( lhs, selStack_.back() ) );
      selStack_.pop_back();
      break;
    }
    case ( kOr ) : {
      SelectorPtr lhs = cut_;
      cut_ = SelectorPtr( new OrCombiner( lhs, selStack_.back() ) );
      selStack_.pop_back();
      break;
    }
    case ( kNot ) : {
      SelectorPtr arg = cut_;
      cut_ = SelectorPtr( new NotCombiner( arg ) );	    
    }
    };
    cmbStack_.pop_back();
  }
#ifdef BOOST_SPIRIT_DEBUG 
  BOOST_SPIRIT_DEBUG_OUT << "cut set" << std::endl;
#endif
}

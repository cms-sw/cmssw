#include "PhysicsTools/Utilities/src/ExpressionSetter.h"
#include "PhysicsTools/Utilities/src/AndCombiner.h"
#include "PhysicsTools/Utilities/src/OrCombiner.h"
#include "PhysicsTools/Utilities/src/NotCombiner.h"
#include "FWCore/Utilities/interface/EDMException.h"
using namespace reco::parser;

void ExpressionSetter::operator()( const char *, const char * ) const {
  if ( exprStack_.size() == 0 ) 
    throw edm::Exception( edm::errors::LogicError )
      << "When trying parse an expression, expression stack is empty!";
  expr_ = exprStack_.back();
}

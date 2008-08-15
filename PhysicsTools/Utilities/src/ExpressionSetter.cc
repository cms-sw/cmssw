#include "PhysicsTools/Utilities/src/ExpressionSetter.h"
#include "PhysicsTools/Utilities/src/AndCombiner.h"
#include "PhysicsTools/Utilities/src/OrCombiner.h"
#include "PhysicsTools/Utilities/src/NotCombiner.h"
#include "PhysicsTools/Utilities/interface/Exception.h"
using namespace reco::parser;

void ExpressionSetter::operator()( const char *begin, const char * ) const {
  if ( exprStack_.size() == 0 ) 
    throw Exception( begin )
      << "Grammar error: When trying parse an expression, expression stack is empty! Please contact a developer.";
  expr_ = exprStack_.back();
}

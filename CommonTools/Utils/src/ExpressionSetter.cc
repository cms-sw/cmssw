#include "CommonTools/Utils/src/ExpressionSetter.h"
#include "CommonTools/Utils/src/AndCombiner.h"
#include "CommonTools/Utils/src/OrCombiner.h"
#include "CommonTools/Utils/src/NotCombiner.h"
#include "CommonTools/Utils/interface/Exception.h"

using namespace reco::parser;

void ExpressionSetter::operator()( const char *begin, const char * ) const {
  if ( exprStack_.size() == 0 ) 
    throw Exception( begin )
      << "Grammar error: When trying parse an expression, expression stack is empty! Please contact a developer.";
  expr_ = exprStack_.back();
}

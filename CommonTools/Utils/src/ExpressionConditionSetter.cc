#include "CommonTools/Utils/src/ExpressionConditionSetter.h"
#include "CommonTools/Utils/src/ExpressionCondition.h"

using namespace reco::parser;

void ExpressionConditionSetter::operator()(const char *, const char *) const {
  ExpressionBase * ep = new ExpressionCondition(expStack_, selStack_);
  ExpressionPtr e(ep);
  expStack_.push_back(e);
}

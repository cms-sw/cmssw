#ifndef HDQM_EXPRESSION_EVALUATOR__INC__
#define HDQM_EXPRESSION_EVALUATOR__INC__


#include <stack>
#include <string>

namespace HDQMExpressionEvaluator
{

  enum
  {
    eval_ok = 0,
    eval_unbalanced,
    eval_invalidoperator,
    eval_invalidoperand,
    eval_evalerr
  };

  int calculateLong(std::string expr, long &r);
  int calculateDouble(std::string expr, double &r);
}

#endif

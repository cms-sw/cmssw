#ifndef CommonTools_Utils_ExpressionQuaterOperator_h
#define CommonTools_Utils_ExpressionQuaterOperator_h
/* \class reco::parser::ExpressionQuaterOperator
 *
 * Quater Operator expression
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted by Luca Lista, INFN
 *
 * \version $Revision: 1.2 $
 *
 */
#include "CommonTools/Utils/src/ExpressionBase.h"
#include "CommonTools/Utils/src/ExpressionStack.h"

namespace reco {
  namespace parser {
    template<typename Op>
    struct ExpressionQuaterOperator : public ExpressionBase {
      virtual double value(const edm::ObjectWithDict& o) const { 
	return op_(args_[0]->value(o), args_[1]->value(o), args_[2]->value(o), args_[3]->value(o));
      }
      ExpressionQuaterOperator(ExpressionStack & expStack) { 
	args_[3] = expStack.back(); expStack.pop_back();
	args_[2] = expStack.back(); expStack.pop_back();
	args_[1] = expStack.back(); expStack.pop_back();
	args_[0] = expStack.back(); expStack.pop_back();
      }
    private:
      Op op_;
      ExpressionPtr args_[4];
    };
  }
}

#endif

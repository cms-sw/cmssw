#ifndef CommonTools_Utils_ExpressionCondition_h
#define CommonTools_Utils_ExpressionCondition_h
/* \class reco::parser::ExpressionCondition
 *
 * Unary Operator expression
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted by Luca Lista, INFN
 *
 * \version $Revision: 1.3 $
 *
 */
#include "CommonTools/Utils/src/ExpressionBase.h"
#include "CommonTools/Utils/src/SelectorBase.h"
#include "CommonTools/Utils/src/ExpressionStack.h"

namespace reco {
  namespace parser {
    struct ExpressionCondition : public ExpressionBase {
      virtual double value(const edm::ObjectWithDict& o) const { 
	return (*cond_)(o) ? true_->value(o) : false_->value(o);
      }
      ExpressionCondition(ExpressionStack & expStack, SelectorStack & selStack) { 
	false_ = expStack.back(); expStack.pop_back();
	true_  = expStack.back(); expStack.pop_back();
	cond_  = selStack.back(); selStack.pop_back();
      }
      private:
      ExpressionPtr true_, false_;
      SelectorPtr cond_;
    };
  }
}

#endif

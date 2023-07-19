#ifndef CommonTools_Utils_ExpressionCondition_h
#define CommonTools_Utils_ExpressionCondition_h
/* \class reco::parser::ExpressionCondition
 *
 * Unary Operator expression
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted by Luca Lista, INFN
 *
 * \version $Revision: 1.2 $
 *
 */
#include "CommonTools/Utils/interface/parser/ExpressionBase.h"
#include "CommonTools/Utils/interface/parser/SelectorBase.h"
#include "CommonTools/Utils/interface/parser/SelectorStack.h"
#include "CommonTools/Utils/interface/parser/ExpressionStack.h"

namespace reco {
  namespace parser {
    struct ExpressionCondition : public ExpressionBase {
      double value(const edm::ObjectWithDict& o) const override {
        return (*cond_)(o) ? true_->value(o) : false_->value(o);
      }
      ExpressionCondition(ExpressionStack& expStack, SelectorStack& selStack) {
        false_ = expStack.back();
        expStack.pop_back();
        true_ = expStack.back();
        expStack.pop_back();
        cond_ = selStack.back();
        selStack.pop_back();
      }

    private:
      ExpressionPtr true_, false_;
      SelectorPtr cond_;
    };
  }  // namespace parser
}  // namespace reco

#endif

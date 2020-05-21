#ifndef CommonTools_Utils_TrinarySelector_h
#define CommonTools_Utils_TrinarySelector_h
/* \class reco::parser::TrinarySelector
 *
 * Trinary selector
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted by Luca Lista, INFN
 *
 * \version $Revision: 1.2 $
 *
 */
#include <utility>

#include "CommonTools/Utils/src/SelectorBase.h"
#include "CommonTools/Utils/src/ExpressionBase.h"
#include "CommonTools/Utils/src/ComparisonBase.h"

namespace reco {
  namespace parser {
    struct TrinarySelector : public SelectorBase {
      TrinarySelector(std::shared_ptr<ExpressionBase> lhs,
                      std::shared_ptr<ComparisonBase> cmp1,
                      std::shared_ptr<ExpressionBase> mid,
                      std::shared_ptr<ComparisonBase> cmp2,
                      std::shared_ptr<ExpressionBase> rhs)
          : lhs_(std::move(lhs)),
            cmp1_(std::move(cmp1)),
            mid_(std::move(mid)),
            cmp2_(std::move(cmp2)),
            rhs_(std::move(rhs)) {}
      bool operator()(const edm::ObjectWithDict& o) const override {
        return cmp1_->compare(lhs_->value(o), mid_->value(o)) && cmp2_->compare(mid_->value(o), rhs_->value(o));
      }
      std::shared_ptr<ExpressionBase> lhs_;
      std::shared_ptr<ComparisonBase> cmp1_;
      std::shared_ptr<ExpressionBase> mid_;
      std::shared_ptr<ComparisonBase> cmp2_;
      std::shared_ptr<ExpressionBase> rhs_;
    };
  }  // namespace parser
}  // namespace reco

#endif

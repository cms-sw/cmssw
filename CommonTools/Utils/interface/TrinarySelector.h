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
#include "CommonTools/Utils/interface/SelectorBase.h"
#include "CommonTools/Utils/interface/ExpressionBase.h"
#include "CommonTools/Utils/interface/ComparisonBase.h"

namespace reco {
  namespace parser {
    struct TrinarySelector : public SelectorBase {
      TrinarySelector(std::shared_ptr<ExpressionBase> lhs,
                      std::shared_ptr<ComparisonBase> cmp1,
                      std::shared_ptr<ExpressionBase> mid,
                      std::shared_ptr<ComparisonBase> cmp2,
                      std::shared_ptr<ExpressionBase> rhs)
          : lhs_(lhs), cmp1_(cmp1), mid_(mid), cmp2_(cmp2), rhs_(rhs) {}
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

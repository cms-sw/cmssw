#ifndef CommonTools_Utils_BinarySelector_h
#define CommonTools_Utils_BinarySelector_h
/* \class reco::parser::BinarySelector
 *
 * Binary selector
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
    struct BinarySelector : public SelectorBase {
      BinarySelector(std::shared_ptr<ExpressionBase> lhs,
                     std::shared_ptr<ComparisonBase> cmp,
                     std::shared_ptr<ExpressionBase> rhs)
          : lhs_(lhs), cmp_(cmp), rhs_(rhs) {}
      bool operator()(const edm::ObjectWithDict& o) const override {
        return cmp_->compare(lhs_->value(o), rhs_->value(o));
      }
      std::shared_ptr<ExpressionBase> lhs_;
      std::shared_ptr<ComparisonBase> cmp_;
      std::shared_ptr<ExpressionBase> rhs_;
    };
  }  // namespace parser
}  // namespace reco

#endif

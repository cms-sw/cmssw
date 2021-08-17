#ifndef CommonTools_Utils_LogicalBinaryOperator_h
#define CommonTools_Utils_LogicalBinaryOperator_h
/* \class reco::parser::LogicalBinaryOperator
 *
 * logical AND combiner
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted by Luca Lista, INFN
 *
 * \version $Revision: 1.2 $
 *
 */
#include "CommonTools/Utils/interface/SelectorBase.h"
#include "CommonTools/Utils/interface/SelectorStack.h"

namespace reco {
  namespace parser {
    template <typename Op>
    struct LogicalBinaryOperator : public SelectorBase {
      LogicalBinaryOperator(SelectorStack &selStack) {
        rhs_ = selStack.back();
        selStack.pop_back();
        lhs_ = selStack.back();
        selStack.pop_back();
      }
      bool operator()(const edm::ObjectWithDict &o) const override;

    private:
      Op op_;
      SelectorPtr lhs_, rhs_;
    };

    template <>
    bool LogicalBinaryOperator<std::logical_and<bool> >::operator()(const edm::ObjectWithDict &o) const;
    template <>
    bool LogicalBinaryOperator<std::logical_or<bool> >::operator()(const edm::ObjectWithDict &o) const;
  }  // namespace parser
}  // namespace reco

#endif

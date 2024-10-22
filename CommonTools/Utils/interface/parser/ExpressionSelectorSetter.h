#ifndef CommonTools_Utils_ExpressionSelectorSetter_h
#define CommonTools_Utils_ExpressionSelectorSetter_h
/* \class reco::parser::ExpressionSelectorSetter
 *
 * Creates an implicit Binary selector setter by comparing an expression to 0
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted by Luca Lista, INFN
 *
 * \version $Revision: 1.2 $
 *
 */
#include "CommonTools/Utils/interface/parser/SelectorStack.h"
#include "CommonTools/Utils/interface/parser/ExpressionStack.h"
#include "CommonTools/Utils/interface/parser/BinarySelector.h"
#include "CommonTools/Utils/interface/parser/ExpressionNumber.h"
#include "CommonTools/Utils/interface/parser/Comparison.h"
#include "CommonTools/Utils/interface/parser/Exception.h"

#include <functional>

namespace reco {
  namespace parser {
    class ExpressionSelectorSetter {
    public:
      ExpressionSelectorSetter(SelectorStack& selStack, ExpressionStack& expStack)
          : selStack_(selStack), expStack_(expStack) {}

      void operator()(const char* begin, const char*) const {
        if (expStack_.empty())
          throw Exception(begin) << "Grammar error: empty expression stack. Please contact developer."
                                 << "\"";
        std::shared_ptr<ExpressionBase> rhs = expStack_.back();
        expStack_.pop_back();
        std::shared_ptr<ExpressionBase> lhs(new ExpressionNumber(0.0));
        std::shared_ptr<ComparisonBase> comp(new Comparison<std::not_equal_to<double> >());
#ifdef BOOST_SPIRIT_DEBUG
        BOOST_SPIRIT_DEBUG_OUT << "pushing expression selector" << std::endl;
#endif
        selStack_.push_back(SelectorPtr(new BinarySelector(lhs, comp, rhs)));
      }

    private:
      SelectorStack& selStack_;
      ExpressionStack& expStack_;
    };
  }  // namespace parser
}  // namespace reco

#endif

#ifndef CommonTools_Utils_BinarySelectorSetter_h
#define CommonTools_Utils_BinarySelectorSetter_h
/* \class reco::parser::BinarySelectorSetter
 *
 * Binary selector setter
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted by Luca Lista, INFN
 *
 * \version $Revision: 1.2 $
 *
 */
#include "CommonTools/Utils/interface/parser/SelectorStack.h"
#include "CommonTools/Utils/interface/parser/ComparisonStack.h"
#include "CommonTools/Utils/interface/parser/ExpressionStack.h"
#include "CommonTools/Utils/interface/parser/BinarySelector.h"
#include "CommonTools/Utils/interface/parser/Exception.h"

namespace reco {
  namespace parser {
    class BinarySelectorSetter {
    public:
      BinarySelectorSetter(SelectorStack& selStack, ComparisonStack& cmpStack, ExpressionStack& expStack)
          : selStack_(selStack), cmpStack_(cmpStack), expStack_(expStack) {}

      void operator()(const char* begin, const char*) const {
        if (expStack_.empty())
          throw Exception(begin) << "Grammar error: empty expression stack. Please contact developer.";
        if (cmpStack_.empty())
          throw Exception(begin) << "Grammar error: empty comparator stack. Please contact developer."
                                 << "\"";
        std::shared_ptr<ExpressionBase> rhs = expStack_.back();
        expStack_.pop_back();
        std::shared_ptr<ExpressionBase> lhs = expStack_.back();
        expStack_.pop_back();
        std::shared_ptr<ComparisonBase> comp = cmpStack_.back();
        cmpStack_.pop_back();
#ifdef BOOST_SPIRIT_DEBUG
        BOOST_SPIRIT_DEBUG_OUT << "pushing binary selector" << std::endl;
#endif
        selStack_.push_back(SelectorPtr(new BinarySelector(lhs, comp, rhs)));
      }

    private:
      SelectorStack& selStack_;
      ComparisonStack& cmpStack_;
      ExpressionStack& expStack_;
    };
  }  // namespace parser
}  // namespace reco

#endif

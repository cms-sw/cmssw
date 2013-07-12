#ifndef CommonTools_Utils_ExpressionConditionSetter_h
#define CommonTools_Utils_ExpressionConditionSetter_h
/* \class reco::parser::ExpressionCondition
 *
 * Numerical expression setter
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.1 $
 *
 */
#include "CommonTools/Utils/src/ExpressionStack.h"
#include "CommonTools/Utils/src/SelectorStack.h"

namespace reco {
  namespace parser {
    struct ExpressionConditionSetter {
      ExpressionConditionSetter(ExpressionStack & expStack, SelectorStack & selStack) : 
	expStack_(expStack), selStack_(selStack) { }
      void operator()(const char *, const char *) const;
    private:
      ExpressionStack & expStack_;
      SelectorStack & selStack_;
    };
  }
}

#endif

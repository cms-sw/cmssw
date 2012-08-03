#ifndef CommonTools_Utils_LogicalUnaryOperator_h
#define CommonTools_Utils_LogicalUnaryOperator_h
/* \class reco::parser::LogicalUnaryOperator
 *
 * logical AND combiner
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted by Luca Lista, INFN
 *
 * \version $Revision: 1.2 $
 *
 */
#include "CommonTools/Utils/src/SelectorBase.h"
#include "CommonTools/Utils/src/SelectorStack.h"

namespace reco {
  namespace parser {    
    template<typename Op>
    struct LogicalUnaryOperator : public SelectorBase {
      LogicalUnaryOperator(SelectorStack & selStack) {
	rhs_ = selStack.back(); selStack.pop_back();
      }
      virtual bool operator()(const edm::ObjectWithDict& o) const {
	return op_((*rhs_)(o));
      }
      private:
      Op op_;
      SelectorPtr rhs_;
    };
  }
}

#endif

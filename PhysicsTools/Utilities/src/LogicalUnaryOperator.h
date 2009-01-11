#ifndef Utilities_LogicalUnaryOperator_h
#define Utilities_LogicalUnaryOperator_h
/* \class reco::parser::LogicalUnaryOperator
 *
 * logical AND combiner
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted to Reflex by Luca Lista, INFN
 *
 * \version $Revision: 1.1 $
 *
 */
#include "PhysicsTools/Utilities/src/SelectorBase.h"
#include "PhysicsTools/Utilities/src/SelectorStack.h"

namespace reco {
  namespace parser {    
    template<typename Op>
    struct LogicalUnaryOperator : public SelectorBase {
      LogicalUnaryOperator(SelectorStack & selStack) {
	rhs_ = selStack.back(); selStack.pop_back();
      }
      virtual bool operator()(const Reflex::Object& o) const {
	return op_((*rhs_)(o));
      }
      private:
      Op op_;
      SelectorPtr rhs_;
    };
  }
}

#endif

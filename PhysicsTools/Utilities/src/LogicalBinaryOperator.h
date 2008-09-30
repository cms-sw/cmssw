#ifndef Utilities_LogicalBinaryOperator_h
#define Utilities_LogicalBinaryOperator_h
/* \class reco::parser::LogicalBinaryOperator
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
    struct LogicalBinaryOperator : public SelectorBase {
      LogicalBinaryOperator(SelectorStack & selStack) {
	rhs_ = selStack.back(); selStack.pop_back();
	lhs_ = selStack.back(); selStack.pop_back();
      }
      virtual bool operator()(const ROOT::Reflex::Object& o) const {
	return op_((*lhs_)(o), (*rhs_)(o));
      }
      private:
      Op op_;
      SelectorPtr lhs_, rhs_;
    };
  }
}

#endif

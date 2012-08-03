#ifndef CommonTools_Utils_CutBinaryOperator_h
#define CommonTools_Utils_CutBinaryOperator_h
/* \class reco::parser::CutBinaryOperator
 *
 * Binary Operator expression
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted by Luca Lista, INFN
 *
 * \version $Revision: 1.2 $
 *
 */
#include "CommonTools/Utils/src/CutBase.h"
#include "CommonTools/Utils/src/CutStack.h"

namespace reco {
  namespace parser {
    template<typename Op>
    struct CutBinaryOperator : public CutBase {
      virtual double value(const edm::ObjectWithDict& o) const { 
	return op_((*lhs_).value(o), (*rhs_).value(o));
      }
      CutBinaryOperator(CutStack & cutStack) { 
	rhs_ = cutStack.back(); cutStack.pop_back();
	lhs_ = cutStack.back(); cutStack.pop_back();
      }
    private:
      Op op_;
      CutPtr lhs_, rhs_;
    };
  }
}

#endif

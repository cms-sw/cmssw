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
#include "CommonTools/Utils/src/SelectorBase.h"
#include "CommonTools/Utils/src/ExpressionBase.h"
#include "CommonTools/Utils/src/ComparisonBase.h"
#include <boost/shared_ptr.hpp>

namespace reco {
  namespace parser {    
    struct TrinarySelector : public SelectorBase {
      TrinarySelector( boost::shared_ptr<ExpressionBase> lhs,
		       boost::shared_ptr<ComparisonBase> cmp1,
		       boost::shared_ptr<ExpressionBase> mid,
		       boost::shared_ptr<ComparisonBase> cmp2,
		       boost::shared_ptr<ExpressionBase> rhs ) :
	lhs_( lhs ), cmp1_( cmp1 ), mid_( mid ), cmp2_( cmp2 ),rhs_( rhs ) {}
      virtual bool operator()( const edm::ObjectWithDict& o ) const {
	return 
	  cmp1_->compare( lhs_->value( o ), mid_->value( o ) ) &&
	  cmp2_->compare( mid_->value( o ), rhs_->value( o ) );
      }
      boost::shared_ptr<ExpressionBase> lhs_;
      boost::shared_ptr<ComparisonBase> cmp1_;
      boost::shared_ptr<ExpressionBase> mid_;
      boost::shared_ptr<ComparisonBase> cmp2_;
      boost::shared_ptr<ExpressionBase> rhs_;
    };
  }
}

#endif

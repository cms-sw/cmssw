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
#include "CommonTools/Utils/src/SelectorBase.h"
#include "CommonTools/Utils/src/ExpressionBase.h"
#include "CommonTools/Utils/src/ComparisonBase.h"
#include <boost/shared_ptr.hpp>

namespace reco {
  namespace parser {
    struct BinarySelector : public SelectorBase {
      BinarySelector( boost::shared_ptr<ExpressionBase> lhs,
		      boost::shared_ptr<ComparisonBase> cmp,
		      boost::shared_ptr<ExpressionBase> rhs ) :
	lhs_( lhs ), cmp_( cmp ), rhs_( rhs ) { }
      virtual bool operator()( const edm::ObjectWithDict & o ) const {
	return cmp_->compare( lhs_->value( o ), rhs_->value( o ) );
      }
      boost::shared_ptr<ExpressionBase> lhs_;
      boost::shared_ptr<ComparisonBase> cmp_;
      boost::shared_ptr<ExpressionBase> rhs_;
    };
  }
}

#endif

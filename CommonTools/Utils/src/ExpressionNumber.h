#ifndef CommonTools_Utils_ExpressionNumber_h
#define CommonTools_Utils_ExpressionNumber_h
/* \class reco::parser::ExpressionNumber
 *
 * Numberical expression
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted by Luca Lista, INFN
 *
 * \version $Revision: 1.3 $
 *
 */
#include "CommonTools/Utils/src/ExpressionBase.h"

namespace reco {
  namespace parser {
    struct ExpressionNumber : public ExpressionBase {
      virtual double value( const edm::ObjectWithDict& ) const { return value_; }
      ExpressionNumber( double value ) : value_( value ) { }
    private:
      double value_;
    };
  }
}

#endif

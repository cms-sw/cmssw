#ifndef Parser_ExpressionPtr_h
#define Parser_ExpressionPtr_h
/* \class reco::parser::ExpressionPtr
 *
 * Shared pointer to Expression
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.2 $
 *
 */
#include <boost/shared_ptr.hpp>

namespace reco {
  namespace parser {
    struct ExpressionBase;
    typedef boost::shared_ptr<ExpressionBase> ExpressionPtr;
  }
}

#endif

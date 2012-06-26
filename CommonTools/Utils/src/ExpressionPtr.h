#ifndef Parser_ExpressionPtr_h
#define Parser_ExpressionPtr_h
/* \class reco::parser::ExpressionPtr
 *
 * Shared pointer to Reflex Expression
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.1 $
 *
 */
#include <boost/shared_ptr.hpp>

namespace reco {
  namespace parser {
    class ExpressionBase;
    typedef boost::shared_ptr<ExpressionBase> ExpressionPtr;
  }
}

#endif

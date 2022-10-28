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

#include <memory>

namespace reco {
  namespace parser {
    struct ExpressionBase;
    typedef std::shared_ptr<ExpressionBase> ExpressionPtr;
  }  // namespace parser
}  // namespace reco

#endif

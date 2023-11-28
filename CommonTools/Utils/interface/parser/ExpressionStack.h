#ifndef Parser_ExpressionStack_h
#define Parser_ExpressionStack_h
/* \class reco::parser::ExpressionStack
 *
 * Stack of parsed expressions
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted by Luca Lista, INFN
 *
 * \version $Revision: 1.2 $
 *
 */

#include <vector>
#include <memory>

namespace reco {
  namespace parser {
    struct ExpressionBase;

    typedef std::vector<std::shared_ptr<ExpressionBase> > ExpressionStack;
  }  // namespace parser
}  // namespace reco

#endif

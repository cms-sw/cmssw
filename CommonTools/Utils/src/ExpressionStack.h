#ifndef Parser_ExpressionStack_h
#define Parser_ExpressionStack_h
/* \class reco::parser::ExpressionStack
 *
 * Stack of parsed expressions
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted by Luca Lista, INFN
 *
 * \version $Revision: 1.3 $
 *
 */
#include <boost/shared_ptr.hpp>
#include <vector>

namespace reco {
  namespace parser {
    struct ExpressionBase;

    typedef std::vector<boost::shared_ptr<ExpressionBase> > ExpressionStack;
  }
}

#endif

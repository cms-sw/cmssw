#ifndef Parser_ExpressionBase_h
#define Parser_ExpressionBase_h
/* \class reco::parser::ExpressionBase
 *
 * Base class for parsed expressions
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted by Luca Lista, INFN
 *
 * \version $Revision: 1.3 $
 *
 */
#include <boost/shared_ptr.hpp>
#include <vector>

namespace edm { class ObjectWithDict; }

namespace reco {
  namespace parser {
    struct ExpressionBase {
      virtual ~ExpressionBase() { }
      virtual double value( const edm::ObjectWithDict & ) const = 0;
    };
    typedef boost::shared_ptr<ExpressionBase> ExpressionPtr;
  }
}

#endif

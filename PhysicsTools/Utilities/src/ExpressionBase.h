#ifndef Parser_ExpressionBase_h
#define Parser_ExpressionBase_h
/* \class reco::parser::ExpressionBase
 *
 * Base class for parsed expressions
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted to Reflex by Luca Lista, INFN
 *
 * \version $Revision: 1.1 $
 *
 */
#include <boost/shared_ptr.hpp>
#include <vector>

namespace ROOT { namespace Reflex { class Object; } }

namespace reco {
  namespace parser {
    struct ExpressionBase {
      virtual ~ExpressionBase() { }
      virtual double value( const ROOT::Reflex::Object & ) const = 0;
    };
    typedef boost::shared_ptr<ExpressionBase> ExpressionPtr;
  }
}

#endif

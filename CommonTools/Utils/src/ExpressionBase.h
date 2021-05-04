#ifndef Parser_ExpressionBase_h
#define Parser_ExpressionBase_h
/* \class reco::parser::ExpressionBase
 *
 * Base class for parsed expressions
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted by Luca Lista, INFN
 *
 */

#include <vector>
#include <memory>

namespace edm {
  class ObjectWithDict;
}

namespace reco {
  namespace parser {
    struct ExpressionBase {
      virtual ~ExpressionBase() {}
      virtual double value(const edm::ObjectWithDict&) const = 0;
    };
    typedef std::shared_ptr<ExpressionBase> ExpressionPtr;
  }  // namespace parser
}  // namespace reco

#endif

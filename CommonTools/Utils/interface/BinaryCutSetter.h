#ifndef CommonTools_Utils_BinaryCutSetter_h
#define CommonTools_Utils_BinaryCutSetter_h
/* \class reco::parser::BinaryCutSetter
 *
 * Cut setter
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted by Luca Lista, INFN
 *
 * \version $Revision: 1.2 $
 *
 */
#include "CommonTools/Utils/interface/SelectorStack.h"
#include "CommonTools/Utils/interface/LogicalBinaryOperator.h"

namespace reco {
  namespace parser {
    template <typename Op>
    struct BinaryCutSetter {
      BinaryCutSetter(SelectorStack& selStack) : selStack_(selStack) {}
      void operator()(const char*, const char*) const {
        selStack_.push_back(SelectorPtr(new LogicalBinaryOperator<Op>(selStack_)));
      }
      void operator()(const char&) const {
        const char* c;
        operator()(c, c);
      }
      SelectorStack& selStack_;
    };
  }  // namespace parser
}  // namespace reco

#endif

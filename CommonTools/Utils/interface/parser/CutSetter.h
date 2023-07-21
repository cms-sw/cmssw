#ifndef CommonTools_Utils_CutSetter_h
#define CommonTools_Utils_CutSetter_h
/* \class reco::parser::CutSetter
 *
 * Cut setter
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted by Luca Lista, INFN
 *
 * \version $Revision: 1.3 $
 *
 */
#include "CommonTools/Utils/interface/parser/SelectorPtr.h"
#include "CommonTools/Utils/interface/parser/SelectorStack.h"
#include <cassert>

namespace reco {
  namespace parser {
    struct CutSetter {
      CutSetter(SelectorPtr& cut, SelectorStack& selStack) : cut_(cut), selStack_(selStack) {}

      void operator()(const char*, const char*) const {
        assert(nullptr == cut_.get());
        assert(!selStack_.empty());
        cut_ = selStack_.back();
        selStack_.pop_back();
      }
      SelectorPtr& cut_;
      SelectorStack& selStack_;
    };
  }  // namespace parser
}  // namespace reco

#endif

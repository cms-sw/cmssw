#ifndef CommonTools_Utils_NotCombiner_h
#define CommonTools_Utils_NotCombiner_h
/* \class reco::parser::NotCombiner
 *
 * logical NOT combiner
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted by Luca Lista, INFN
 *
 * \version $Revision: 1.2 $
 *
 */
#include "CommonTools/Utils/interface/parser/SelectorBase.h"
#include "CommonTools/Utils/interface/parser/SelectorPtr.h"

namespace reco {
  namespace parser {
    struct NotCombiner : public SelectorBase {
      NotCombiner(SelectorPtr arg) : arg_(arg) {}
      bool operator()(const edm::ObjectWithDict& o) const override { return !(*arg_)(o); }

    private:
      SelectorPtr arg_;
    };
  }  // namespace parser
}  // namespace reco

#endif

#ifndef CommonTools_Utils_AndCombiner_h
#define CommonTools_Utils_AndCombiner_h
/* \class reco::parser::AndCombiner
 *
 * logical AND combiner
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
    struct AndCombiner : public SelectorBase {
      AndCombiner(SelectorPtr lhs, SelectorPtr rhs) : lhs_(lhs), rhs_(rhs) {}
      bool operator()(const edm::ObjectWithDict& o) const override { return (*lhs_)(o) && (*rhs_)(o); }

    private:
      SelectorPtr lhs_, rhs_;
    };
  }  // namespace parser
}  // namespace reco

#endif

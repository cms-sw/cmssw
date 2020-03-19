#ifndef Parser_SelectorPtr_h
#define Parser_SelectorPtr_h
/* \class reco::parser::SelectorPtr
 *
 * Shared pointer to selector
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted by Luca Lista, INFN
 *
 * \version $Revision: 1.2 $
 *
 */

#include <memory>

namespace reco {
  namespace parser {
    class SelectorBase;
    typedef std::shared_ptr<SelectorBase> SelectorPtr;
  }  // namespace parser
}  // namespace reco

#endif

#ifndef Parser_SelectorPtr_h
#define Parser_SelectorPtr_h
/* \class reco::parser::SelectorPtr
 *
 * Shared pointer to Reflex selector
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted to Reflex by Luca Lista, INFN
 *
 * \version $Revision: 1.1 $
 *
 */
#include <boost/shared_ptr.hpp>

namespace reco {
  namespace parser {
    class SelectorBase;
    typedef boost::shared_ptr<SelectorBase> SelectorPtr;
  }
}

#endif

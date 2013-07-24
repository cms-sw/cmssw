#ifndef Parser_SelectorPtr_h
#define Parser_SelectorPtr_h
/* \class reco::parser::SelectorPtr
 *
 * Shared pointer to selector
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted by Luca Lista, INFN
 *
 * \version $Revision: 1.3 $
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

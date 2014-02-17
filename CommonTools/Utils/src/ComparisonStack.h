#ifndef Parser_ComparisonStack_h
#define Parser_ComparisonStack_h
/* \class reco::parser::ComparisonStack
 *
 * Comparison stack
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted by Luca Lista, INFN
 *
 * \version $Revision: 1.3 $
 *
 */
#include <boost/shared_ptr.hpp>
#include <vector>

namespace reco {
  namespace parser {
    class ComparisonBase;
    typedef std::vector<boost::shared_ptr<ComparisonBase> > ComparisonStack;
  }
}

#endif

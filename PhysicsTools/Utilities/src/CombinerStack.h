#ifndef Utilities_CombinerStack_h
#define Utilities_CombinerStack_h
/* \class reco::parser::CombinerStack
 *
 * Combiner stack
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted to Reflex by Luca Lista, INFN
 *
 * \version $Revision: 1.1 $
 *
 */
#include "PhysicsTools/Utilities/src/Combiner.h"
#include <vector>

namespace reco {
  namespace parser {    
    typedef std::vector<Combiner> CombinerStack;
  }
}

#endif

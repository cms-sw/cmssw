#ifndef UtilAlgos_CutParser_h
#define UtilAlgos_CutParser_h
/* \class CutParser
 *
 *  Cut string parser
 *
 * \author Luca Lista, INFN, 
 *         from original implementation by Chris Jones
 *
 */
#include "PhysicsTools/Utilities/interface/SelectorBase.h"
#include "PhysicsTools/UtilAlgos/interface/ObjectMethodSet.h"
#include <string>
#include <boost/shared_ptr.hpp>
#include <map>

namespace reco {
  namespace cutparser {
    class ObjectSelectorBase {
    };

    static bool parse( const std::string &,
		       const methods::methodMap & ,
		       boost::shared_ptr<ObjectSelectorBase> & ) {
    }
  }
}
#endif

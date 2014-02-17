#ifndef CommonTools_Utils_SelectorBase_h
#define CommonTools_Utils_SelectorBase_h
/** \class SelectorBase
 *
 * Base class for all object selector 
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.3 $
 *
 */

namespace edm {class ObjectWithDict;}

namespace reco {
  namespace parser {
    class SelectorBase {
    public:
      /// destructor
      virtual ~SelectorBase() { }
      /// return true if the object is selected
      virtual bool operator()(const edm::ObjectWithDict & c) const = 0;
    };
  }
}

#endif

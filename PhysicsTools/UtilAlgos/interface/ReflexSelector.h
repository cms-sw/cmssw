#ifndef UtilAlgos_ReflexSelector_h
#define UtilAlgos_ReflexSelector_h
/** \class ReflexSelector
 *
 * Base class for all Reflex object selector 
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.3 $
 *
 * $Id: ReflexSelector.h,v 1.3 2006/03/03 10:09:18 llista Exp $
 *
 */

namespace ROOT {
  namespace Reflex {
    class Object;
  }
}

class ReflexSelector {
public:
  /// destructor
  virtual ~ReflexSelector() { }
  /// return true if the Refle object is selected
  virtual bool operator()( const ROOT::Reflex::Object & c ) const = 0;
};

#endif

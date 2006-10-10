#ifndef Utilities_ReflexSelector_h
#define Utilities_ReflexSelector_h
/** \class ReflexSelector
 *
 * Base class for all Reflex object selector 
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.1 $
 *
 * $Id: ReflexSelector.h,v 1.1 2006/07/27 16:34:43 llista Exp $
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

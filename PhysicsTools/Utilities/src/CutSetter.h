#ifndef Utilities_CutSetter_h
#define Utilities_CutSetter_h
/* \class reco::parser::CutSetter
 *
 * Cut setter
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted to Reflex by Luca Lista, INFN
 *
 * \version $Revision: 1.3 $
 *
 */
#include "PhysicsTools/Utilities/src/SelectorPtr.h"
#include "PhysicsTools/Utilities/src/SelectorStack.h"
#include "PhysicsTools/Utilities/src/CombinerStack.h"

namespace reco {
  namespace parser {    
    struct CutSetter {
      CutSetter( SelectorPtr & cut, SelectorStack & selStack, CombinerStack & cmbStack ) :
	cut_( cut ), selStack_( selStack ), cmbStack_( cmbStack ) { }
      
      void operator()( const char*, const char* ) const;
      SelectorPtr & cut_;
      SelectorStack & selStack_;
      CombinerStack & cmbStack_;
    };
  }
 }

#endif

#ifndef DTSegment_DTRecSegment4DAlgoFactory_h
#define DTSegment_DTRecSegment4DAlgoFactory_h

/** \class DTRecSegment4DAlgoFactory
 *
 *  Factory of seal plugins for DT 4D segments reconstruction algorithms.
 *  The plugins are concrete implementations of DTRecSegment4DBaseAlgo base class.
 *
 * $Date:  $
 * $Revision:  $
 * \author Riccardo Bellan - INFN TO <riccardo.bellan@cern.ch>
 *
 */

#include <PluginManager/PluginFactory.h>
#include "RecoLocalMuon/DTSegment/src/DTRecSegment4DBaseAlgo.h"


// C++ Headers

// ======================================================================

// Class DTRecSegment4DAlgoFactory Interface

class DTRecSegment4DAlgoFactory : 
public seal::PluginFactory<DTRecSegment4DBaseAlgo*(const edm::ParameterSet&)> {

  public:

    /// Constructor
    DTRecSegment4DAlgoFactory() ;

    /// Destructor
    virtual ~DTRecSegment4DAlgoFactory() ;

    // Operations
    static DTRecSegment4DAlgoFactory* get(void);

  private:
    static DTRecSegment4DAlgoFactory s_instance;

};
#endif 

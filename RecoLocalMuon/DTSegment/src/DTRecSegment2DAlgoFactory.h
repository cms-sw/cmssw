#ifndef DTSegment_DTRecSegment2DAlgoFactory_h
#define DTSegment_DTRecSegment2DAlgoFactory_h

/** \class DTRecSegment2DAlgoFactory
 *
 *  Factory of seal plugins for DT 2D segments reconstruction algorithms.
 *  The plugins are concrete implementations of DTRecSegment2DBaseAlgo base class.
 *
 * $Date: 01/03/2006 16:44:00 CET $
 * $Revision: 1.0 $
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 *
 */

/* Base Class Headers */
#include <PluginManager/PluginFactory.h>
#include "RecoLocalMuon/DTSegment/src/DTRecSegment2DBaseAlgo.h"

/* Collaborating Class Declarations */

/* C++ Headers */

/* ====================================================================== */

/* Class DTRecSegment2DAlgoFactory Interface */

class DTRecSegment2DAlgoFactory : 
public seal::PluginFactory<DTRecSegment2DBaseAlgo*(const edm::ParameterSet&)> {

  public:

    /// Constructor
    DTRecSegment2DAlgoFactory() ;

    /// Destructor
    virtual ~DTRecSegment2DAlgoFactory() ;

/* Operations */ 
    static DTRecSegment2DAlgoFactory* get(void);

  private:
    static DTRecSegment2DAlgoFactory s_instance;

};
#endif // DTSegment_DTRecSegment2DAlgoFactory_h

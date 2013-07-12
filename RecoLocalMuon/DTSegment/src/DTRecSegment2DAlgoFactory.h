#ifndef DTSegment_DTRecSegment2DAlgoFactory_h
#define DTSegment_DTRecSegment2DAlgoFactory_h

/** \class DTRecSegment2DAlgoFactory
 *
 *  Factory of seal plugins for DT 2D segments reconstruction algorithms.
 *  The plugins are concrete implementations of DTRecSegment2DBaseAlgo base class.
 *
 * $Date: 2007/04/10 23:14:36 $
 * $Revision: 1.2 $
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 *
 */

/* Base Class Headers */
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "RecoLocalMuon/DTSegment/src/DTRecSegment2DBaseAlgo.h"

/* Collaborating Class Declarations */

/* C++ Headers */

/* ====================================================================== */

/* Class DTRecSegment2DAlgoFactory Interface */

typedef edmplugin::PluginFactory<DTRecSegment2DBaseAlgo *(const edm::ParameterSet &)> DTRecSegment2DAlgoFactory;
#endif // DTSegment_DTRecSegment2DAlgoFactory_h

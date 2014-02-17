#ifndef DTSegment_DTRecSegment4DAlgoFactory_h
#define DTSegment_DTRecSegment4DAlgoFactory_h

/** \class DTRecSegment4DAlgoFactory
 *
 *  Factory of seal plugins for DT 4D segments reconstruction algorithms.
 *  The plugins are concrete implementations of DTRecSegment4DBaseAlgo base class.
 *
 * $Date: 2007/04/17 22:46:49 $
 * $Revision: 1.3 $
 * \author Riccardo Bellan - INFN TO <riccardo.bellan@cern.ch>
 *
 */

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "RecoLocalMuon/DTSegment/src/DTRecSegment4DBaseAlgo.h"


// C++ Headers

// ======================================================================

// Class DTRecSegment4DAlgoFactory Interface

typedef edmplugin::PluginFactory<DTRecSegment4DBaseAlgo *(const edm::ParameterSet &)> DTRecSegment4DAlgoFactory;
#endif 

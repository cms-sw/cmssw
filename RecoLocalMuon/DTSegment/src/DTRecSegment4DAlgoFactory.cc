/** \class DTRecSegment4DAlgoFactory
 *
 *  Factory of seal plugins for DT 4D segments reconstruction algorithms.
 *  The plugins are concrete implementations of DTRecSegment4DBaseAlgo base class.
 *
 * $Date: 2006/05/15 09:36:59 $
 * $Revision: 1.2 $
 * \author Riccardo Bellan - INFN TO <riccardo.bellan@cern.ch>
 */

// This Class Header
#include "RecoLocalMuon/DTSegment/src/DTRecSegment4DAlgoFactory.h"

/* Collaborating Class Header */

/* C++ Headers */

#include "FWCore/PluginManager/interface/PluginFactory.h"

EDM_REGISTER_PLUGINFACTORY(DTRecSegment4DAlgoFactory,"DTRecSegment4DAlgoFactory");


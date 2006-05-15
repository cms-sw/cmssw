/** \class DTRecSegment4DAlgoFactory
 *
 *  Factory of seal plugins for DT 4D segments reconstruction algorithms.
 *  The plugins are concrete implementations of DTRecSegment4DBaseAlgo base class.
 *
 * $Date: 2006/04/19 14:59:33 $
 * $Revision: 1.1 $
 * \author Riccardo Bellan - INFN TO <riccardo.bellan@cern.ch>
 */

// This Class Header
#include "RecoLocalMuon/DTSegment/src/DTRecSegment4DAlgoFactory.h"

/* Collaborating Class Header */

/* C++ Headers */
#include <iostream>
using namespace std;

/* ====================================================================== */
DTRecSegment4DAlgoFactory DTRecSegment4DAlgoFactory::s_instance;

/// Constructor
DTRecSegment4DAlgoFactory::DTRecSegment4DAlgoFactory() :
  seal::PluginFactory<DTRecSegment4DBaseAlgo*(const edm::ParameterSet&)>("DTRecSegment4DAlgoFactory"){}

/// Destructor
DTRecSegment4DAlgoFactory::~DTRecSegment4DAlgoFactory() {
}

DTRecSegment4DAlgoFactory* DTRecSegment4DAlgoFactory::get(void) {
  return &s_instance;
}

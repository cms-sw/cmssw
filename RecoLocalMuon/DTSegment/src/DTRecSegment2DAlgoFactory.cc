/** \file
 *
 * $Date: 2006/03/30 16:53:18 $
 * $Revision: 1.1 $
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 */

/* This Class Header */
#include "RecoLocalMuon/DTSegment/src/DTRecSegment2DAlgoFactory.h"

/* Collaborating Class Header */

/* C++ Headers */
#include <iostream>
using namespace std;

/* ====================================================================== */
DTRecSegment2DAlgoFactory DTRecSegment2DAlgoFactory::s_instance;

/// Constructor
DTRecSegment2DAlgoFactory::DTRecSegment2DAlgoFactory() :
  seal::PluginFactory<DTRecSegment2DBaseAlgo*(const edm::ParameterSet&)>("DTRecSegment2DAlgoFactory"){}

/// Destructor
DTRecSegment2DAlgoFactory::~DTRecSegment2DAlgoFactory() {
}

/* Operations */ 
DTRecSegment2DAlgoFactory* DTRecSegment2DAlgoFactory::get(void) {
  return &s_instance;
}

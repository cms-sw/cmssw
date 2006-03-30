/** \file
 *
 * $Date:  01/03/2006 16:46:15 CET $
 * $Revision: 1.0 $
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
  seal::PluginFactory<DTRecSegment2DBaseAlgo*(const edm::ParameterSet&)>("DTRecSegment2DAlgoFactory"){
    cout << "s_instance " << &s_instance << endl;
}

/// Destructor
DTRecSegment2DAlgoFactory::~DTRecSegment2DAlgoFactory() {
}

/* Operations */ 
DTRecSegment2DAlgoFactory* DTRecSegment2DAlgoFactory::get(void) {
  return &s_instance;
}

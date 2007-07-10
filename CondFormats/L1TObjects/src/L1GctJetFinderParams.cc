#include "CondFormats/L1TObjects/interface/L1GctJetFinderParams.h"

L1GctJetFinderParams::L1GctJetFinderParams(unsigned cJetSeed,
                                           unsigned fJetSeed,
                                           unsigned tJetSeed,
                                           unsigned etaBoundary) :
  CENTRAL_JET_SEED(cJetSeed),
  FORWARD_JET_SEED(fJetSeed),
  TAU_JET_SEED(tJetSeed),
  CENTRAL_FORWARD_ETA_BOUNDARY(etaBoundary) {}

L1GctJetFinderParams::~L1GctJetFinderParams() {}

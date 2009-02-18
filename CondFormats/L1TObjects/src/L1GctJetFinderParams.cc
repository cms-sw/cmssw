#include "CondFormats/L1TObjects/interface/L1GctJetFinderParams.h"

L1GctJetFinderParams::L1GctJetFinderParams(double rgnEtLsb,
					   double htLsb,
					   double cJetSeed,
					   double fJetSeed,
					   double tJetSeed,
					   double tauIsoEtThresh,
					   double htJetEtThresh,
					   double mhtJetEtThresh,
					   unsigned etaBoundary) :
  rgnEtLsb_(rgnEtLsb),
  htLsb_(htLsb),
  cenJetEtSeed_(cJetSeed),
  forJetEtSeed_(fJetSeed),
  tauJetEtSeed_(tJetSeed),
  tauIsoEtThreshold_(tauIsoEtThresh),
  htJetEtThreshold_(htJetEtThresh),
  mhtJetEtThreshold_(mhtJetEtThresh),
  cenForJetEtaBoundary_(etaBoundary)
{ }

L1GctJetFinderParams::~L1GctJetFinderParams() {}

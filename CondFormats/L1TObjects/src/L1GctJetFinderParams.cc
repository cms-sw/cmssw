#include "CondFormats/L1TObjects/interface/L1GctJetFinderParams.h"

L1GctJetFinderParams::L1GctJetFinderParams() :
  rgnEtLsb_(0.),
  htLsb_(0.),
  cenJetEtSeed_(0.),
  forJetEtSeed_(0.),
  tauJetEtSeed_(0.),
  tauIsoEtThreshold_(0.),
  htJetEtThreshold_(0.),
  mhtJetEtThreshold_(0.),
  cenForJetEtaBoundary_(0.)
{ }

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

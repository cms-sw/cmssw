
#ifndef GCTDIGICOLLECTION_H
#define GCTDIGICOLLECTION_H

#include <vector>

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctDigis.h"

typedef std::vector<L1GctEmCand> L1GctIsoEmCollection;
typedef std::vector<L1GctEmCand> L1GctNonIsoEmCollection;
typedef std::vector<L1GctForJet> L1GctForJetCollection;
typedef std::vector<L1GctCenJet> L1GctCenJetCollection;
typedef std::vector<L1GctTauJet> L1GctTauJetCollection;

#endif

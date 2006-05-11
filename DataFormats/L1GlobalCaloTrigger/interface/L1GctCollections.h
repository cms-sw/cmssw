
#ifndef GCTDIGICOLLECTION_H
#define GCTDIGICOLLECTION_H

#include <vector>

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCandidates.h"

typedef std::vector<L1GctIsoEmCand> L1GctIsoEmCollection;
typedef std::vector<L1GctNonIsoEmCand> L1GctNonIsoEmCollection;
typedef std::vector<L1GctForJetCand> L1GctForJetCollection;
typedef std::vector<L1GctCenJetCand> L1GctCenJetCollection;
typedef std::vector<L1GctTauJetCand> L1GctTauJetCollection;

#endif

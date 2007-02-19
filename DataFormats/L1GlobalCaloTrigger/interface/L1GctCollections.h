
#ifndef GCTCOLLECTIONS_H
#define GCTCOLLECTIONS_H

#include <vector>

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctInternEmCand.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEmCand.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCand.h"

typedef std::vector<L1GctInternEmCand> L1GctInternEmCandCollection;
typedef std::vector<L1GctEmCand> L1GctEmCandCollection;
typedef std::vector<L1GctJetCand> L1GctJetCandCollection;


#endif

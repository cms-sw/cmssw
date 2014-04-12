
#ifndef GCTCOLLECTIONS_H
#define GCTCOLLECTIONS_H

#include <vector>

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctInternEmCand.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctInternJetData.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctInternEtSum.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctInternHFData.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctInternHtMiss.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctFibreWord.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEmCand.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCand.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtSums.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCounts.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctHFRingEtSums.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctHFBitCounts.h"

typedef std::vector<L1GctInternEmCand> L1GctInternEmCandCollection;
typedef std::vector<L1GctInternJetData> L1GctInternJetDataCollection;
typedef std::vector<L1GctInternEtSum> L1GctInternEtSumCollection;
typedef std::vector<L1GctInternHFData> L1GctInternHFDataCollection;
typedef std::vector<L1GctInternHtMiss> L1GctInternHtMissCollection;
typedef std::vector<L1GctFibreWord> L1GctFibreCollection;

typedef std::vector<L1GctEmCand> L1GctEmCandCollection;
typedef std::vector<L1GctJetCand> L1GctJetCandCollection;

typedef std::vector<L1GctEtHad> L1GctEtHadCollection;
typedef std::vector<L1GctEtMiss> L1GctEtMissCollection;
typedef std::vector<L1GctEtTotal> L1GctEtTotalCollection;
typedef std::vector<L1GctHtMiss> L1GctHtMissCollection;
typedef std::vector<L1GctJetCounts> L1GctJetCountsCollection;
typedef std::vector<L1GctHFRingEtSums> L1GctHFRingEtSumsCollection;
typedef std::vector<L1GctHFBitCounts> L1GctHFBitCountsCollection;

#endif

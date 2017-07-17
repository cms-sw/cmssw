// -*- C++ -*-
//
// Package:     JetMETCorrections/Modules
// Class  :     JetCorrectorProducers
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Sun, 31 Aug 2014 20:58:29 GMT
//

// system include files

// user include files
#include "FWCore/Framework/interface/MakerMacros.h"
#include "JetCorrectorProducer.h"
#include "JetMETCorrections/Algorithms/interface/LXXXCorrectorImpl.h"
#include "JetMETCorrections/Algorithms/interface/L1OffsetCorrectorImpl.h"
#include "JetMETCorrections/Algorithms/interface/L1JPTOffsetCorrectorImpl.h"
#include "JetMETCorrections/Algorithms/interface/L1FastjetCorrectorImpl.h"
#include "JetMETCorrections/Algorithms/interface/L6SLBCorrectorImpl.h"

typedef JetCorrectorProducer<LXXXCorrectorImpl> LXXXCorrectorProducer;
DEFINE_FWK_MODULE(LXXXCorrectorProducer);

typedef JetCorrectorProducer<L1OffsetCorrectorImpl> L1OffsetCorrectorProducer;
DEFINE_FWK_MODULE(L1OffsetCorrectorProducer);

typedef JetCorrectorProducer<L1JPTOffsetCorrectorImpl> L1JPTOffsetCorrectorProducer;
DEFINE_FWK_MODULE(L1JPTOffsetCorrectorProducer);

typedef JetCorrectorProducer<L1FastjetCorrectorImpl> L1FastjetCorrectorProducer;
DEFINE_FWK_MODULE(L1FastjetCorrectorProducer);

typedef JetCorrectorProducer<L6SLBCorrectorImpl> L6SLBCorrectorProducer;
DEFINE_FWK_MODULE(L6SLBCorrectorProducer);

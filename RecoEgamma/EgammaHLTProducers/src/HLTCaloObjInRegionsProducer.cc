#include "RecoEgamma/EgammaHLTProducers/interface/HLTCaloObjInRegionsProducer.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
using HLTHcalDigisInRegionsProducer = HLTCaloObjInRegionsProducer<HBHEDataFrame>;
DEFINE_FWK_MODULE(HLTHcalDigisInRegionsProducer);

#include "DataFormats/HcalDigi/interface/QIE11DataFrame.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
using HLTHcalQIE11DigisInRegionsProducer = HLTCaloObjInRegionsProducer<QIE11DataFrame, QIE11DigiCollection>;
DEFINE_FWK_MODULE(HLTHcalQIE11DigisInRegionsProducer);

#include "DataFormats/HcalDigi/interface/QIE10DataFrame.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
using HLTHcalQIE10DigisInRegionsProducer = HLTCaloObjInRegionsProducer<QIE10DataFrame, QIE10DigiCollection>;
DEFINE_FWK_MODULE(HLTHcalQIE10DigisInRegionsProducer);

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
using HLTEcalEBDigisInRegionsProducer = HLTCaloObjInRegionsProducer<EBDataFrame, EBDigiCollection>;
DEFINE_FWK_MODULE(HLTEcalEBDigisInRegionsProducer);
using HLTEcalEEDigisInRegionsProducer = HLTCaloObjInRegionsProducer<EEDataFrame, EEDigiCollection>;
DEFINE_FWK_MODULE(HLTEcalEEDigisInRegionsProducer);
using HLTEcalESDigisInRegionsProducer = HLTCaloObjInRegionsProducer<ESDataFrame, ESDigiCollection>;
DEFINE_FWK_MODULE(HLTEcalESDigisInRegionsProducer);

//these two classes are intended to ultimately replace the EcalRecHit and EcalUncalibratedRecHit
//instances of HLTRecHitInAllL1RegionsProducer, particulary as we're free of legacy / stage-1 L1 now
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
using HLTEcalRecHitsInRegionsProducer = HLTCaloObjInRegionsProducer<EcalRecHit>;
DEFINE_FWK_MODULE(HLTEcalRecHitsInRegionsProducer);
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
using HLTEcalUnCalibRecHitsInRegionsProducer = HLTCaloObjInRegionsProducer<EcalUncalibratedRecHit>;
DEFINE_FWK_MODULE(HLTEcalUnCalibRecHitsInRegionsProducer);

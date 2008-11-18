import FWCore.ParameterSet.Config as cms

#------------------------------------------------
#AlCaReco filtering for HCAL isotrk:
#------------------------------------------------

from Configuration.StandardSequences.Reconstruction_cff import *

import EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi
isoTrSiPixelDigis = EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi.siPixelDigis.clone()

from RecoLocalTracker.SiStripZeroSuppression.SiStripZeroSuppression_cfi import *
import EventFilter.SiStripRawToDigi.SiStripDigis_cfi
isoTrSiStripDigis = EventFilter.SiStripRawToDigi.SiStripDigis_cfi.siStripDigis.clone()

import EventFilter.EcalRawToDigiDev.EcalUnpackerData_cfi
isoTrEcalDigis = EventFilter.EcalRawToDigiDev.EcalUnpackerData_cfi.ecalEBunpacker.clone()

import EventFilter.ESRawToDigi.esRawToDigi_cfi
isoTrEcalPreshowerDigis = EventFilter.ESRawToDigi.esRawToDigi_cfi.esRawToDigi.clone()

import EventFilter.HcalRawToDigi.HcalRawToDigi_cfi
isoTrHcalDigis = EventFilter.HcalRawToDigi.HcalRawToDigi_cfi.hcalDigis.clone()

isoTrSiPixelDigis.InputLabel = 'hltSubdetFED'
isoTrSiStripDigis.ProductLabel = 'hltSiStripRegFED'
isoTrEcalDigis.DoRegional = True
isoTrEcalDigis.InputLabel = 'hltEcalRegFED'
isoTrEcalDigis.FedLabel = 'hltEcalRegFED'
isoTrEcalPreshowerDigis.sourceTag = 'hltEcalRegFED'
isoTrHcalDigis.InputLabel = 'hltSubdetFED'

siPixelClusters.src = 'isoTrSiPixelDigis'
siStripZeroSuppression.RawDigiProducersList = cms.VPSet(
    cms.PSet(RawDigiProducer = cms.string('isoTrSiStripDigis'), RawDigiLabel = cms.string('VirginRaw')),
    cms.PSet(RawDigiProducer = cms.string('isoTrSiStripDigis'), RawDigiLabel = cms.string('ProcessedRaw')),
    cms.PSet(RawDigiProducer = cms.string('isoTrSiStripDigis'), RawDigiLabel = cms.string('ScopeMode'))
     )
siStripClusters.DigiProducersList = cms.VPSet(cms.PSet(
    DigiLabel = cms.string('ZeroSuppressed'),
    DigiProducer = cms.string('isoTrSiStripDigis')
),
    cms.PSet(
        DigiLabel = cms.string('VirginRaw'),
        DigiProducer = cms.string('siStripZeroSuppression')
    ),
    cms.PSet(
        DigiLabel = cms.string('ProcessedRaw'),
        DigiProducer = cms.string('siStripZeroSuppression')
    ),
    cms.PSet(
        DigiLabel = cms.string('ScopeMode'),
        DigiProducer = cms.string('siStripZeroSuppression')
    ))
ecalWeightUncalibRecHit.EBdigiCollection = 'isoTrEcalDigis:ebDigis'
ecalWeightUncalibRecHit.EEdigiCollection = 'isoTrEcalDigis:eeDigis'
ecalPreshowerRecHit.ESdigiCollection = 'isoTrEcalPreshowerDigis'
hbhereco.digiLabel = 'isoTrHcalDigis'
horeco.digiLabel = 'isoTrHcalDigis'
hfreco.digiLabel = 'isoTrHcalDigis'

from Calibration.HcalAlCaRecoProducers.alcaisotrk_cfi import *
from Calibration.HcalAlCaRecoProducers.isoHLT_cfi import *

doIsoTrDigi = cms.Sequence(isoTrSiPixelDigis+isoTrSiStripDigis+isoTrEcalDigis+isoTrEcalPreshowerDigis+isoTrHcalDigis)
doLocalReco = cms.Sequence(trackerlocalreco+calolocalreco)
doGlobalReco = cms.Sequence(offlineBeamSpot+recopixelvertexing*ckftracks)
seqALCARECOHcalCalIsoTrk = cms.Sequence(isoHLT*doIsoTrDigi*doLocalReco*doGlobalReco*IsoProd)





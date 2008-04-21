import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaHLTProducers.ecalRegionalEgammaFEDs_cfi import *
from RecoEgamma.EgammaHLTProducers.ecalRegionalMuonsFEDs_cfi import *
from RecoEgamma.EgammaHLTProducers.ecalRegionalTausFEDs_cfi import *
from RecoEgamma.EgammaHLTProducers.ecalRegionalJetsFEDs_cfi import *
from RecoEgamma.EgammaHLTProducers.ecalRegionalRestFEDs_cfi import *
import copy
from EventFilter.EcalRawToDigiDev.EcalUnpackerData_cfi import *
ecalRegionalEgammaDigis = copy.deepcopy(ecalEBunpacker)
import copy
from EventFilter.EcalRawToDigiDev.EcalUnpackerData_cfi import *
ecalRegionalMuonsDigis = copy.deepcopy(ecalEBunpacker)
import copy
from EventFilter.EcalRawToDigiDev.EcalUnpackerData_cfi import *
ecalRegionalTausDigis = copy.deepcopy(ecalEBunpacker)
import copy
from EventFilter.EcalRawToDigiDev.EcalUnpackerData_cfi import *
ecalRegionalJetsDigis = copy.deepcopy(ecalEBunpacker)
import copy
from EventFilter.EcalRawToDigiDev.EcalUnpackerData_cfi import *
ecalRegionalRestDigis = copy.deepcopy(ecalEBunpacker)
from Geometry.CaloEventSetup.CaloGeometry_cfi import *
import copy
from RecoLocalCalo.EcalRecProducers.ecalWeightUncalibRecHit_cfi import *
ecalRegionalEgammaWeightUncalibRecHit = copy.deepcopy(ecalWeightUncalibRecHit)
import copy
from RecoLocalCalo.EcalRecProducers.ecalRecHit_cfi import *
ecalRegionalEgammaRecHitTmp = copy.deepcopy(ecalRecHit)
import copy
from RecoEgamma.EgammaHLTProducers.ecalRecHitMerger_cfi import *
ecalRegionalEgammaRecHit = copy.deepcopy(ecalRecHitMerger)
import copy
from RecoLocalCalo.EcalRecProducers.ecalWeightUncalibRecHit_cfi import *
ecalRegionalMuonsWeightUncalibRecHit = copy.deepcopy(ecalWeightUncalibRecHit)
import copy
from RecoLocalCalo.EcalRecProducers.ecalRecHit_cfi import *
ecalRegionalMuonsRecHitTmp = copy.deepcopy(ecalRecHit)
import copy
from RecoEgamma.EgammaHLTProducers.ecalRecHitMerger_cfi import *
ecalRegionalMuonsRecHit = copy.deepcopy(ecalRecHitMerger)
import copy
from RecoLocalCalo.EcalRecProducers.ecalWeightUncalibRecHit_cfi import *
ecalRegionalTausWeightUncalibRecHit = copy.deepcopy(ecalWeightUncalibRecHit)
import copy
from RecoLocalCalo.EcalRecProducers.ecalRecHit_cfi import *
ecalRegionalTausRecHitTmp = copy.deepcopy(ecalRecHit)
import copy
from RecoEgamma.EgammaHLTProducers.ecalRecHitMerger_cfi import *
ecalRegionalTausRecHit = copy.deepcopy(ecalRecHitMerger)
import copy
from RecoLocalCalo.EcalRecProducers.ecalWeightUncalibRecHit_cfi import *
ecalRegionalJetsWeightUncalibRecHit = copy.deepcopy(ecalWeightUncalibRecHit)
import copy
from RecoLocalCalo.EcalRecProducers.ecalRecHit_cfi import *
ecalRegionalJetsRecHitTmp = copy.deepcopy(ecalRecHit)
import copy
from RecoEgamma.EgammaHLTProducers.ecalRecHitMerger_cfi import *
ecalRegionalJetsRecHit = copy.deepcopy(ecalRecHitMerger)
import copy
from RecoLocalCalo.EcalRecProducers.ecalWeightUncalibRecHit_cfi import *
ecalRegionalRestWeightUncalibRecHit = copy.deepcopy(ecalWeightUncalibRecHit)
import copy
from RecoLocalCalo.EcalRecProducers.ecalRecHit_cfi import *
ecalRegionalRestRecHitTmp = copy.deepcopy(ecalRecHit)
import copy
from RecoEgamma.EgammaHLTProducers.ecalRecHitMerger_cfi import *
ecalRecHitAll = copy.deepcopy(ecalRecHitMerger)
from RecoLocalCalo.EcalRecProducers.ecalPreshowerRecHit_cfi import *
ecalRegionalEgammaRecoSequence = cms.Sequence(ecalRegionalEgammaFEDs*ecalRegionalEgammaDigis*ecalRegionalEgammaWeightUncalibRecHit*ecalRegionalEgammaRecHitTmp*ecalRegionalEgammaRecHit+ecalPreshowerRecHit)
ecalRegionalMuonsRecoSequence = cms.Sequence(ecalRegionalMuonsFEDs*ecalRegionalMuonsDigis*ecalRegionalMuonsWeightUncalibRecHit*ecalRegionalMuonsRecHitTmp*ecalRegionalMuonsRecHit+ecalPreshowerRecHit)
ecalRegionalTausRecoSequence = cms.Sequence(ecalRegionalTausFEDs*ecalRegionalTausDigis*ecalRegionalTausWeightUncalibRecHit*ecalRegionalTausRecHitTmp*ecalRegionalTausRecHit+ecalPreshowerRecHit)
ecalRegionalJetsRecoSequence = cms.Sequence(ecalRegionalJetsFEDs*ecalRegionalJetsDigis*ecalRegionalJetsWeightUncalibRecHit*ecalRegionalJetsRecHitTmp*ecalRegionalJetsRecHit+ecalPreshowerRecHit)
ecalAllRecoSequence = cms.Sequence(ecalRegionalRestFEDs*ecalRegionalRestDigis*ecalRegionalRestWeightUncalibRecHit*ecalRegionalRestRecHitTmp*ecalRecHitAll+ecalPreshowerRecHit)
ecalRegionalEgammaDigis.InputLabel = 'rawDataCollector'
ecalRegionalEgammaDigis.DoRegional = True
ecalRegionalEgammaDigis.FedLabel = 'ecalRegionalEgammaFEDs'
ecalRegionalMuonsDigis.InputLabel = 'rawDataCollector'
ecalRegionalMuonsDigis.DoRegional = True
ecalRegionalMuonsDigis.FedLabel = 'ecalRegionalMuonsFEDs'
ecalRegionalTausDigis.InputLabel = 'rawDataCollector'
ecalRegionalTausDigis.DoRegional = True
ecalRegionalTausDigis.FedLabel = 'ecalRegionalTausFEDs'
ecalRegionalJetsDigis.InputLabel = 'rawDataCollector'
ecalRegionalJetsDigis.DoRegional = True
ecalRegionalJetsDigis.FedLabel = 'ecalRegionalJetsFEDs'
ecalRegionalRestDigis.InputLabel = 'rawDataCollector'
ecalRegionalRestDigis.DoRegional = True
ecalRegionalRestDigis.FedLabel = 'ecalRegionalRestFEDs'
ecalRegionalEgammaWeightUncalibRecHit.EBdigiCollection = cms.InputTag("ecalRegionalEgammaDigis","ebDigis")
ecalRegionalEgammaWeightUncalibRecHit.EEdigiCollection = cms.InputTag("ecalRegionalEgammaDigis","eeDigis")
ecalRegionalEgammaRecHitTmp.EBuncalibRecHitCollection = cms.InputTag("ecalRegionalEgammaWeightUncalibRecHit","EcalUncalibRecHitsEB")
ecalRegionalEgammaRecHitTmp.EEuncalibRecHitCollection = cms.InputTag("ecalRegionalEgammaWeightUncalibRecHit","EcalUncalibRecHitsEE")
ecalRegionalMuonsWeightUncalibRecHit.EBdigiCollection = cms.InputTag("ecalRegionalMuonsDigis","ebDigis")
ecalRegionalMuonsWeightUncalibRecHit.EEdigiCollection = cms.InputTag("ecalRegionalMuonsDigis","eeDigis")
ecalRegionalMuonsRecHitTmp.EBuncalibRecHitCollection = cms.InputTag("ecalRegionalMuonsWeightUncalibRecHit","EcalUncalibRecHitsEB")
ecalRegionalMuonsRecHitTmp.EEuncalibRecHitCollection = cms.InputTag("ecalRegionalMuonsWeightUncalibRecHit","EcalUncalibRecHitsEE")
ecalRegionalTausWeightUncalibRecHit.EBdigiCollection = cms.InputTag("ecalRegionalTausDigis","ebDigis")
ecalRegionalTausWeightUncalibRecHit.EEdigiCollection = cms.InputTag("ecalRegionalTausDigis","eeDigis")
ecalRegionalTausRecHitTmp.EBuncalibRecHitCollection = cms.InputTag("ecalRegionalTausWeightUncalibRecHit","EcalUncalibRecHitsEB")
ecalRegionalTausRecHitTmp.EEuncalibRecHitCollection = cms.InputTag("ecalRegionalTausWeightUncalibRecHit","EcalUncalibRecHitsEE")
ecalRegionalJetsWeightUncalibRecHit.EBdigiCollection = cms.InputTag("ecalRegionalJetsDigis","ebDigis")
ecalRegionalJetsWeightUncalibRecHit.EEdigiCollection = cms.InputTag("ecalRegionalJetsDigis","eeDigis")
ecalRegionalJetsRecHitTmp.EBuncalibRecHitCollection = cms.InputTag("ecalRegionalJetsWeightUncalibRecHit","EcalUncalibRecHitsEB")
ecalRegionalJetsRecHitTmp.EEuncalibRecHitCollection = cms.InputTag("ecalRegionalJetsWeightUncalibRecHit","EcalUncalibRecHitsEE")
ecalRegionalRestWeightUncalibRecHit.EBdigiCollection = cms.InputTag("ecalRegionalRestDigis","ebDigis")
ecalRegionalRestWeightUncalibRecHit.EEdigiCollection = cms.InputTag("ecalRegionalRestDigis","eeDigis")
ecalRegionalRestRecHitTmp.EBuncalibRecHitCollection = cms.InputTag("ecalRegionalRestWeightUncalibRecHit","EcalUncalibRecHitsEB")
ecalRegionalRestRecHitTmp.EEuncalibRecHitCollection = cms.InputTag("ecalRegionalRestWeightUncalibRecHit","EcalUncalibRecHitsEE")


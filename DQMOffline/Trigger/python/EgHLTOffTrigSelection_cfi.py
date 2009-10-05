import FWCore.ParameterSet.Config as cms


from DQMOffline.Trigger.EgHLTOffEleSelection_cfi import *
from DQMOffline.Trigger.EgHLTOffPhoSelection_cfi import *

egHLTOffBaseEleTrigCuts = cms.PSet (
    trigName = cms.string("default"),                
    barrel = cms.PSet(egHLTOffEleBarrelCuts),
    endcap = cms.PSet(egHLTOffEleEndcapCuts)
)

egHLTOffBasePhoTrigCuts = cms.PSet (
    trigName = cms.string("default"),                
    barrel = cms.PSet(egHLTOffPhoBarrelCuts),
    endcap = cms.PSet(egHLTOffPhoEndcapCuts)
)

egHLTOffEleEt10SWCuts = cms.PSet(egHLTOffBaseEleTrigCuts)
egHLTOffEleEt10SWCuts.trigName = "hltL1NonIsoHLTNonIsoSingleElectronEt10PixelMatchFilter"
egHLTOffEleEt10SWCuts.barrel.minEt = 10.
egHLTOffEleEt10SWCuts.barrel.cuts = "et"
egHLTOffEleEt10SWCuts.endcap.minEt = 10.
egHLTOffEleEt10SWCuts.endcap.cuts = "et"

egHLTOffEleEt15SWCuts = cms.PSet(egHLTOffBaseEleTrigCuts)
egHLTOffEleEt15SWCuts.trigName = "hltL1NonIsoHLTNonIsoSingleElectronEt15PixelMatchFilter"
egHLTOffEleEt15SWCuts.barrel.minEt = 15.
egHLTOffEleEt15SWCuts.barrel.cuts = "et"
egHLTOffEleEt15SWCuts.endcap.minEt = 15.
egHLTOffEleEt15SWCuts.endcap.cuts = "et"

egHLTOffEleEt20SWCuts  = cms.PSet(egHLTOffBaseEleTrigCuts)
egHLTOffEleEt20SWCuts.trigName = "hltL1NonIsoHLTNonIsoSingleElectronEt20PixelMatchFilter"
egHLTOffEleEt20SWCuts.barrel.minEt = 20.
egHLTOffEleEt20SWCuts.barrel.cuts = "et"
egHLTOffEleEt20SWCuts.endcap.minEt = 20.
egHLTOffEleEt20SWCuts.endcap.cuts = "et"

egHLTOffEleEt15SWEleIdCuts = cms.PSet(egHLTOffBaseEleTrigCuts)
egHLTOffEleEt15SWEleIdCuts.trigName = "hltL1NonIsoHLTNonIsoSingleElectronEt15EleIdDphiFilter"
egHLTOffEleEt15SWEleIdCuts.barrel.minEt = 15.
egHLTOffEleEt15SWEleIdCuts.barrel.maxSigmaIEtaIEta = 0.015
egHLTOffEleEt15SWEleIdCuts.barrel.maxDEtaIn = 0.008
egHLTOffEleEt15SWEleIdCuts.barrel.maxDPhiIn = 0.1
egHLTOffEleEt15SWEleIdCuts.barrel.cuts = "et:dEtaIn:dPhiIn:sigmaIEtaIEta:ctfTrack"
egHLTOffEleEt15SWEleIdCuts.endcap.minEt = 15.
egHLTOffEleEt15SWEleIdCuts.endcap.maxSigmaIEtaIEta = 0.04
egHLTOffEleEt15SWEleIdCuts.endcap.maxDEtaIn = 0.008
egHLTOffEleEt15SWEleIdCuts.endcap.maxDPhiIn = 0.1
egHLTOffEleEt15SWEleIdCuts.endcap.cuts = "et:dEtaIn:dPhiIn:sigmaIEtaIEta:ctfTrack"

#I am confused by this trigger, it claims to have an et cut of 25 but as far as I can tell its 15
egHLTOffEleEt15SWEleIdLTICuts = cms.PSet(egHLTOffEleEt15SWEleIdCuts)
egHLTOffEleEt15SWEleIdLTICuts.trigName = "hltL1NonIsoHLTNonIsoSingleElectronEt15EleIdTrackIsolFilterESet25LTI"
egHLTOffEleEt15SWEleIdLTICuts.barrel.minEt = 15.
egHLTOffEleEt15SWEleIdLTICuts.barrel.maxHLTIsolTrksEleOverPt=0.5
egHLTOffEleEt15SWEleIdLTICuts.barrel.maxHLTIsolTrksEle=8
egHLTOffEleEt15SWEleIdLTICuts.barrel.cuts = "et:dEtaIn:dPhiIn:sigmaIEtaIEta:hltIsolTrksEle:ctfTrack"
egHLTOffEleEt15SWEleIdLTICuts.endcap.minEt = 15.
egHLTOffEleEt15SWEleIdLTICuts.endcap.maxHLTIsolTrksEleOverPt=0.5
egHLTOffEleEt15SWEleIdLTICuts.endcap.maxHLTIsolTrksEle=8
egHLTOffEleEt15SWEleIdLTICuts.endcap.cuts = "et:dEtaIn:dPhiIn:sigmaIEtaIEta:hltIsolTrksEle:ctfTrack"

egHLTOffEleEt15SWLTICuts = cms.PSet(egHLTOffBaseEleTrigCuts)
egHLTOffEleEt15SWLTICuts.trigName = "hltL1NonIsoHLTNonIsoSingleElectronEt15LTITrackIsolFilter"
egHLTOffEleEt15SWLTICuts.barrel.minEt = 15.
egHLTOffEleEt15SWLTICuts.barrel.maxHLTIsolTrksEleOverPt=0.5
egHLTOffEleEt15SWLTICuts.barrel.maxHLTIsolTrksEle=8
egHLTOffEleEt15SWLTICuts.barrel.cuts = "et:hltIsolTrksEle:ctfTrack"
egHLTOffEleEt15SWLTICuts.endcap.minEt = 15.
egHLTOffEleEt15SWLTICuts.endcap.maxHLTIsolTrksEleOverPt=0.5
egHLTOffEleEt15SWLTICuts.endcap.maxHLTIsolTrksEle=8
egHLTOffEleEt15SWLTICuts.endcap.cuts = "et:hltIsolTrksEle:ctfTrack"

egHLTOffDoubleEleEt10SWCuts = cms.PSet(egHLTOffBaseEleTrigCuts)
egHLTOffDoubleEleEt10SWCuts.trigName = "hltL1NonIsoHLTNonIsoDoubleElectronEt10PixelMatchFilter"
egHLTOffDoubleEleEt10SWCuts.barrel.minEt = 10.
egHLTOffDoubleEleEt10SWCuts.barrel.cuts = "et"
egHLTOffDoubleEleEt10SWCuts.endcap.minEt = 10.
egHLTOffDoubleEleEt10SWCuts.endcap.cuts = "et"


egHLTOffPhoEt10Cuts = cms.PSet(egHLTOffBasePhoTrigCuts)
egHLTOffPhoEt10Cuts.trigName = "hltL1NonIsoHLTNonIsoSinglePhotonEt10HcalIsolFilter"
egHLTOffPhoEt10Cuts.barrel.minEt = 10.
egHLTOffPhoEt10Cuts.barrel.cuts = "et"
egHLTOffPhoEt10Cuts.endcap.minEt = 10.
egHLTOffPhoEt10Cuts.endcap.cuts = "et"

egHLTOffPhoEt15Cuts = cms.PSet(egHLTOffBasePhoTrigCuts)
egHLTOffPhoEt15Cuts.trigName = "hltL1NonIsoHLTNonIsoSinglePhotonEt15HcalIsolFilter"
egHLTOffPhoEt15Cuts.barrel.minEt = 15.
egHLTOffPhoEt15Cuts.barrel.cuts = "et"
egHLTOffPhoEt15Cuts.endcap.minEt = 15.
egHLTOffPhoEt15Cuts.endcap.cuts = "et"

egHLTOffPhoEt25Cuts = cms.PSet(egHLTOffBasePhoTrigCuts)
egHLTOffPhoEt25Cuts.trigName = "hltL1NonIsoHLTNonIsoSinglePhotonEt25HcalIsolFilter"
egHLTOffPhoEt25Cuts.barrel.minEt = 25.
egHLTOffPhoEt25Cuts.barrel.cuts = "et"
egHLTOffPhoEt25Cuts.endcap.minEt = 25.
egHLTOffPhoEt25Cuts.endcap.cuts = "et"

egHLTOffPhoEt25LEITICuts = cms.PSet(egHLTOffBasePhoTrigCuts)
egHLTOffPhoEt25LEITICuts.trigName = "hltL1NonIsoHLTLEITISinglePhotonEt25TrackIsolFilter"
egHLTOffPhoEt25LEITICuts.barrel.minEt = 25.
egHLTOffPhoEt25LEITICuts.barrel.maxHLTIsolTrksPho = 0.
egHLTOffPhoEt25LEITICuts.barrel.maxHLTIsolTrksPhoOverPt = 0.05
egHLTOffPhoEt25LEITICuts.barrel.cuts = "et"
egHLTOffPhoEt25LEITICuts.endcap.minEt = 25.
egHLTOffPhoEt25LEITICuts.endcap.maxHLTIsolTrksPho = 0.
egHLTOffPhoEt25LEITICuts.endcap.maxHLTIsolTrksPhoOverPt = 0.05
egHLTOffPhoEt25LEITICuts.endcap.cuts = "et"


egHLTOffPhoEt30Cuts = cms.PSet(egHLTOffBasePhoTrigCuts)
egHLTOffPhoEt30Cuts.trigName = "hltL1NonIsoHLTNonIsoSinglePhotonEt30HcalIsolFilter"
egHLTOffPhoEt30Cuts.barrel.minEt = 30.
egHLTOffPhoEt30Cuts.barrel.cuts = "et"
egHLTOffPhoEt30Cuts.endcap.minEt = 30.
egHLTOffPhoEt30Cuts.endcap.cuts = "et"



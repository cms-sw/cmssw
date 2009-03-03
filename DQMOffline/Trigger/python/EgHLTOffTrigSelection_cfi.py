import FWCore.ParameterSet.Config as cms


from DQMOffline.Trigger.EgHLTOffEleSelection_cfi import *
from DQMOffline.Trigger.EgHLTOffPhoSelection_cfi import *

egHLTOffBaseEleTrigCuts = cms.PSet (
    trigName = cms.string("default"),                
    barrel = cms.PSet(egHLTOffEleBarrelCuts),
    endcap = cms.PSet(egHLTOffEleEndcapCuts)
)

egHLTOffEleEt15Cuts = cms.PSet(egHLTOffBaseEleTrigCuts)
egHLTOffEleEt15Cuts.trigName = "hltL1NonIsoHLTNonIsoSingleElectronEt15TrackIsolFilter"
egHLTOffEleEt15Cuts.barrel.minEt = 15.
egHLTOffEleEt15Cuts.endcap.minEt = 15.

egHLTOffEleLWEt15Cuts = cms.PSet(egHLTOffBaseEleTrigCuts);
egHLTOffEleLWEt15Cuts.trigName = "hltL1NonIsoHLTNonIsoSingleElectronLWEt15TrackIsolFilter"
egHLTOffEleLWEt15Cuts.barrel.minEt = 15.
egHLTOffEleLWEt15Cuts.endcap.minEt = 15.

egHLTOffDoubleEleEt5Cuts = cms.PSet(egHLTOffBaseEleTrigCuts)
egHLTOffDoubleEleEt5Cuts.trigName = "hltL1NonIsoHLTNonIsoDoubleElectronEt5TrackIsolFilter"
egHLTOffDoubleEleEt5Cuts.barrel.minEt = 5.
egHLTOffDoubleEleEt5Cuts.barrel.cuts= "et:detEta:hltIsolHad:hltIsolTrksEle"
egHLTOffDoubleEleEt5Cuts.endcap.minEt = 5.
egHLTOffDoubleEleEt5Cuts.endcap.cuts= "et:detEta:hltIsolHad:hltIsolTrksEle"

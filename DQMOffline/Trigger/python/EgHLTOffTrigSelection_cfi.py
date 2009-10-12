import FWCore.ParameterSet.Config as cms


from DQMOffline.Trigger.EgHLTOffEleSelection_cfi import *
from DQMOffline.Trigger.EgHLTOffPhoSelection_cfi import *

egHLTOffBaseEleTrigCuts = cms.PSet (
    trigName = cms.string("default"),                
    barrel = cms.PSet(egHLTOffEleBarrelCuts),
    endcap = cms.PSet(egHLTOffEleEndcapCuts)
)




egHLTOffEleEt20Cuts  = cms.PSet(egHLTOffBaseEleTrigCuts)
egHLTOffEleEt20Cuts.trigName = "hltL1NonIsoHLTNonIsoSingleElectronEt20PixelMatchFilter"
egHLTOffEleEt20Cuts.barrel.minEt = 20.
egHLTOffEleEt20Cuts.barrel.cuts = "et"
egHLTOffEleEt20Cuts.endcap.minEt = 20.
egHLTOffEleEt20Cuts.endcap.cuts = "et"

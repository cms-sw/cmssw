import FWCore.ParameterSet.Config as cms

#Define the HLT path to be used.
import HLTrigger.HLTfilters.hltHighLevel_cfi as hlt

exoticaHPTEHLT = hlt.hltHighLevel.clone()
exoticaHPTEHLT.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
exoticaHPTEHLT.HLTPaths = ['HLT_DoubleEle33_v*', 'HLT_DoubleEle33_CaloIdL_v*', 'HLT_DoubleEle33_CaloIdT_v*', 'HLT_DoubleEle45_CaloIdL_v*', 'HLT_DoubleEle33_CaloIdL_CaloIsoT_v*', 'HLT_DoublePhoton33_*' , 'HLT_Photon225_NoHE_v*','HLT_Photon200_NoHE_v*','HLT_Photon26_Photon18_v*','HLT_Photon125_v*','HLT_Photon135_v*']
exoticaHPTEHLT.throw = cms.bool( False )


#Define the Reco quality cut
exoticaRecoDiHPTEFilter = cms.EDFilter(
    "PtMinGsfElectronCountFilter",
    src = cms.InputTag("gedGsfElectrons"),
    ptMin = cms.double(30.0),
    minNumber = cms.uint32(2)
    )

#
exoDiHPTESequence = cms.Sequence(
    exoticaHPTEHLT+exoticaRecoDiHPTEFilter


)


import FWCore.ParameterSet.Config as cms

OutALCARECOEcalCalElectron = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOEcalCalElectron')
    ),
    outputCommands = cms.untracked.vstring('drop  *', 
        'keep  *_electronFilter_*_*', 
        'keep  *_alCaIsolatedElectrons_*_*', 
        'keep recoCaloMETs_met_*_*',
        'keep edmTriggerResults_TriggerResults__*', 
        'keep edmHepMCProduct_*_*_*',                                   

#        'keep *_egammaEcalIsolation_*_*', 
#        'keep *_egammaEcalRelIsolation_*_*', 
#        'keep *_egammaElectronSqPtTkIsolation_*_*', 
#        'keep *_egammaElectronTkIsolation_*_*',
#        'keep *_egammaElectronTkNumIsolation_*_*', 
#        'keep *_egammaElectronTkRelIsolation_*_*', 
#        'keep *_egammaHOETower_*_*', 
#        'keep *_egammaHOE_*_*', 
#        'keep *_egammaHcalIsolation_*_*', 
#        'keep *_egammaTowerIsolation_*_*',
   

        'keep *_eleIsoFromDepsTk_*_*',                       
        'keep *_eleIsoFromDepsEcalFromHits_*_*',                       
        'keep *_eleIsoFromDepsHcalFromTowers_*_*',                       
        'keep *_eleIsoFromDepsHcalDepth1FromTowers_*_*',                       
        'keep *_eleIsoFromDepsHcalDepth2FromTowers_*_*',                       


        'keep *_MEtoEDMConverter_*_*')
)


import FWCore.ParameterSet.Config as cms

##_____________________________________________________________ AOD content __||
RecoMETAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoCaloMETs_caloMet_*_*',
                                           'keep recoCaloMETs_caloMetBE_*_*',
                                           'keep recoCaloMETs_caloMetBEFO_*_*',
                                           'keep recoCaloMETs_caloMetM_*_*',
                                           'keep recoPFMETs_pfMet_*_*',
                                           'keep recoPFMETs_pfChMet_*_*',
                                           'keep floatedmValueMap_puppiNoLep_*_*',
                                           'keep recoPFMETs_pfMetPuppi_*_*',
                                           'keep recoMuonMETCorrectionDataedmValueMap_muonMETValueMapProducer_*_*',
                                          # 'drop recoHcalNoiseRBXs_*_*_*',
                                           'keep HcalNoiseSummary_hcalnoise_*_*',
                                           #'keep *GlobalHaloData_*_*_*',
                                           'keep recoGlobalHaloData_GlobalHaloData_*_*',
                                           'keep recoCSCHaloData_CSCHaloData_*_*',
                                           'keep recoBeamHaloSummary_BeamHaloSummary_*_*'
                                           )
    )

RecoGenMETAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoGenMETs_*_*_*')
    )

RecoHcalNoiseAOD = cms.PSet(
    outputCommands = cms.untracked.vstring( # 'drop recoHcalNoiseRBXs_hcalnoise_*_*',
                                           'keep HcalNoiseSummary_hcalnoise_*_*'
                                           )
    )

##____________________________________________________________ RECO content __||
RecoMETRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoHcalNoiseRBXs_hcalnoise_*_*',
                                           #'keep *HaloData_*_*_*',
                                           'keep recoEcalHaloData_EcalHaloData_*_*',
                                           'keep recoHcalHaloData_HcalHaloData_*_*'
                                           )
    )
RecoMETRECO.outputCommands.extend(RecoMETAOD.outputCommands)

RecoGenMETRECO = cms.PSet(
    outputCommands = cms.untracked.vstring()
    )
RecoGenMETRECO.outputCommands.extend(RecoGenMETAOD.outputCommands)

RecoHcalNoiseRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoHcalNoiseRBXs_hcalnoise_*_*')
    )
RecoHcalNoiseRECO.outputCommands.extend(RecoHcalNoiseAOD.outputCommands)

##______________________________________________________ Full Event content __||
RecoMETFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *HaloData_*_*_*',
                                           'keep *BeamHaloSummary_BeamHaloSummary_*_*'
                                           )
    )
RecoMETFEVT.outputCommands.extend(RecoMETRECO.outputCommands)

RecoGenMETFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring()
    )
RecoGenMETFEVT.outputCommands.extend(RecoGenMETRECO.outputCommands)

RecoHcalNoiseFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring()
    )
RecoHcalNoiseFEVT.outputCommands.extend(RecoHcalNoiseRECO.outputCommands)
##____________________________________________________________________________||

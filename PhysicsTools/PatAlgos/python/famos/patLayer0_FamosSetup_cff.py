import FWCore.ParameterSet.Config as cms

# FAMOS replace statements ###
def setup(process) :
    process.allLayer0Muons.muonSource = cms.InputTag('paramMuons','ParamGlobalMuons')
    process.corMetType1Icone5Muons.inputMuonsLabel = cms.InputTag('paramMuons', 'ParamGlobalMuons')   ## muons to be used for MET corrections
    process.patAODMuonIsolations.collection   = cms.InputTag('paramMuons','ParamGlobalMuons')
    process.patAODMuonIsolations.associations = cms.VInputTag(
            cms.InputTag('muParamGlobalIsoDepositCalByAssociatorTowers','ecal'),
            cms.InputTag('muParamGlobalIsoDepositCalByAssociatorTowers','hcal'),
            cms.InputTag('muParamGlobalIsoDepositCalByAssociatorTowers','ho'),
            cms.InputTag('muParamGlobalIsoDepositTk'),
            cms.InputTag('muParamGlobalIsoDepositJets')
            )
    process.layer0MuonIsolations.associations = cms.VInputTag(
            cms.InputTag('muParamGlobalIsoDepositCalByAssociatorTowers','ecal'),
            cms.InputTag('muParamGlobalIsoDepositCalByAssociatorTowers','hcal'),
            cms.InputTag('muParamGlobalIsoDepositCalByAssociatorTowers','ho'),
            cms.InputTag('muParamGlobalIsoDepositTk'),
            cms.InputTag('muParamGlobalIsoDepositJets')
            )
    process.allLayer0Muons.isolation.tracker.src = cms.InputTag('patAODMuonIsolations','muParamGlobalIsoDepositTk')
    process.allLayer0Muons.isolation.ecal.src    = cms.InputTag('patAODMuonIsolations','muParamGlobalIsoDepositCalByAssociatorTowersecal')
    process.allLayer0Muons.isolation.hcal.src    = cms.InputTag('patAODMuonIsolations','muParamGlobalIsoDepositCalByAssociatorTowershcal')
    process.allLayer0Muons.isolation.user = cms.VPSet(
            cms.PSet(
                src = cms.InputTag("patAODMuonIsolations","muParamGlobalIsoDepositCalByAssociatorTowersho"),
                deltaR = cms.double(0.3),
                cut = cms.double(2.0)
                ), 
            cms.PSet(
                src = cms.InputTag("patAODMuonIsolations","muParamGlobalIsoDepositJets"),
                deltaR = cms.double(0.5),
                cut = cms.double(2.0)
                )
            )


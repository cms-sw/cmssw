import FWCore.ParameterSet.Config as cms

# FAMOS replace statements ###

def setup(process) :
    # to use Famos muons
    process.allLayer1Muons.isolation.tracker.src = cms.InputTag('layer0MuonIsolations','muParamGlobalIsoDepositTk')
    process.allLayer1Muons.isolation.ecal.src    = cms.InputTag('layer0MuonIsolations','muParamGlobalIsoDepositCalByAssociatorTowersecal')
    process.allLayer1Muons.isolation.hcal.src    = cms.InputTag('layer0MuonIsolations','muParamGlobalIsoDepositCalByAssociatorTowershcal')
    process.allLayer1Muons.isolation.user        = cms.VPSet()
    process.allLayer1Muons.isoDeposits.tracker = cms.InputTag('layer0MuonIsolations','muParamGlobalIsoDepositTk')
    process.allLayer1Muons.isoDeposits.ecal    = cms.InputTag('layer0MuonIsolations','muParamGlobalIsoDepositCalByAssociatorTowersecal')
    process.allLayer1Muons.isoDeposits.hcal    = cms.InputTag('layer0MuonIsolations','muParamGlobalIsoDepositCalByAssociatorTowershcal')
    process.allLayer1Muons.isoDeposits.user    = cms.VInputTag( cms.InputTag('layer0MuonIsolations','muParamGlobalIsoDepositCalByAssociatorTowersho') )
    
    # De-activate trigger matching until we manage to produce it in FastSim
    process.allLayer1Electrons.addTrigMatch = False
    process.allLayer1Muons.addTrigMatch     = False
    process.allLayer1Jets.addTrigMatch      = False
    process.allLayer1METs.addTrigMatch      = False
    process.allLayer1Photons.addTrigMatch   = False
    process.allLayer1Taus.addTrigMatch      = False

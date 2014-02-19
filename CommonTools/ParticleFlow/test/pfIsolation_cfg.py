
import FWCore.ParameterSet.Config as cms

process = cms.Process("PFISO")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_7_0_0_pre11/RelValProdTTbar/GEN-SIM-RECO/START70_V4-v1/00000/0EA82C3C-646A-E311-9CB3-0025905A6070.root'))


# Tae Jeong, could you please remove these from the PF2PAT sequence?
# they are not used, and are creating problems in such kinds of manipulations:
# process.pfElectronIsolationSequence.remove( process.pfElectronIsoDepositsSequence )
# process.pfElectronIsolationSequence.remove( process.pfElectronIsolationFromDepositsSequence )

# process.load("CommonTools.ParticleFlow.PFBRECO_cff")

from CommonTools.ParticleFlow.Tools.pfIsolation import setupPFElectronIso, setupPFMuonIso, setupPFPhotonIso
process.eleIsoSequence = setupPFElectronIso(process, 'gsfElectrons')
process.muIsoSequence = setupPFMuonIso(process, 'muons')
process.phoIsoSequence = setupPFPhotonIso(process, 'photons')

process.TFileService = cms.Service("TFileService", fileName = cms.string("histo.root") )

process.pfIsoReader = cms.EDAnalyzer("PFIsoReaderDemo",
                                     Electrons = cms.InputTag('gsfElectrons'),
                                     Photons = cms.InputTag('photons'),
                                     PFCandidateMap = cms.InputTag('particleFlow:electrons'),
                                     PrintElectrons = cms.bool(True),
                                     PrintPhotons = cms.bool(True),
                                     IsoDepElectron = cms.VInputTag(cms.InputTag('elPFIsoDepositChargedPFIso'),
                                                                    cms.InputTag('elPFIsoDepositGammaPFIso'),
                                                                    cms.InputTag('elPFIsoDepositNeutralPFIso')),
                                     IsoValElectronPF = cms.VInputTag(cms.InputTag('elPFIsoValueCharged03PFIdPFIso'),
                                                                     cms.InputTag('elPFIsoValueGamma03PFIdPFIso'),
                                                                     cms.InputTag('elPFIsoValueNeutral03PFIdPFIso')),
                                     IsoDepPhoton = cms.VInputTag(cms.InputTag('phPFIsoDepositChargedPFIso'),
                                                                  cms.InputTag('phPFIsoDepositGammaPFIso'),
                                                                  cms.InputTag('phPFIsoDepositNeutralPFIso')),
                                     IsoValPhoton = cms.VInputTag(cms.InputTag('phPFIsoValueCharged03PFIdPFIso'),
                                                                  cms.InputTag('phPFIsoValueGamma03PFIdPFIso'),
                                                                  cms.InputTag('phPFIsoValueNeutral03PFIdPFIso'))

)

process.p = cms.Path(
    # process.pfNoPileUpSequence +
    process.pfParticleSelectionSequence +
    process.eleIsoSequence +
    process.muIsoSequence+
    process.phoIsoSequence+
    process.pfIsoReader
    )


# output ------------------------------------------------------------

process.out = cms.OutputModule("PoolOutputModule",
                               outputCommands = cms.untracked.vstring('keep *'),
                               fileName = cms.untracked.string('pfIsolation.root')
)

process.outpath = cms.EndPath(
    process.out
    )


# other stuff

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 10



import FWCore.ParameterSet.Config as cms

process = cms.Process("PFISO")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
 
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
'/store/relval/CMSSW_5_1_2/RelValZEE/GEN-SIM-RECO/PU_START50_V15A-v1/0003/C61C71DC-8061-E111-AAEB-0025B32036D2.root',
'/store/relval/CMSSW_5_1_2/RelValZEE/GEN-SIM-RECO/PU_START50_V15A-v1/0003/C49AC4E5-7F61-E111-8FB2-003048F1C836.root',
'/store/relval/CMSSW_5_1_2/RelValZEE/GEN-SIM-RECO/PU_START50_V15A-v1/0003/C27CBF93-7E61-E111-AB30-003048F024C2.root',
'/store/relval/CMSSW_5_1_2/RelValZEE/GEN-SIM-RECO/PU_START50_V15A-v1/0003/B46B6A67-8D61-E111-82C0-003048F118AA.root',
'/store/relval/CMSSW_5_1_2/RelValZEE/GEN-SIM-RECO/PU_START50_V15A-v1/0003/36B93F53-7F61-E111-AA27-002481E0D524.root'
    ))
   

# Tae Jeong, could you please remove these from the PF2PAT sequence?
# they are not used, and are creating problems in such kinds of manipulations:
# process.pfElectronIsolationSequence.remove( process.pfElectronIsoDepositsSequence )
# process.pfElectronIsolationSequence.remove( process.pfElectronIsolationFromDepositsSequence )

# process.load("CommonTools.ParticleFlow.PFBRECO_cff")

from CommonTools.ParticleFlow.Tools.pfIsolation import setupPFElectronIso, setupPFMuonIso
process.eleIsoSequence = setupPFElectronIso(process, 'gsfElectrons')
process.muIsoSequence = setupPFMuonIso(process, 'muons')

process.TFileService = cms.Service("TFileService", fileName = cms.string("histo.root") )

process.elePFIsoReader = cms.EDAnalyzer("PFIsoReaderDemo",
                                        Electrons = cms.InputTag('gsfElectrons'),
                                        PFCandidateMap = cms.InputTag('particleFlow:electrons'),
                                        IsoDepElectron = cms.VInputTag(cms.InputTag('elPFIsoDepositChargedPFIso'),
                                                                       cms.InputTag('elPFIsoDepositGammaPFIso'),
                                                                       cms.InputTag('elPFIsoDepositNeutralPFIso')),
                                        IsoValElectronPF = cms.VInputTag(cms.InputTag('elPFIsoValueCharged03PFIdPFIso'),
                                                                         cms.InputTag('elPFIsoValueGamma03PFIdPFIso'),
                                                                         cms.InputTag('elPFIsoValueNeutral03PFIdPFIso')),
                                        IsoValElectronNoPF = cms.VInputTag(cms.InputTag('elPFIsoValueCharged03NoPFIdPFIso'),
                                                                           cms.InputTag('elPFIsoValueGamma03NoPFIdPFIso'),
                                                                           cms.InputTag('elPFIsoValueNeutral03NoPFIdPFIso'))
                                        )

process.p = cms.Path(
    # process.pfNoPileUpSequence +
    process.pfParticleSelectionSequence + 
    process.eleIsoSequence + 
    process.muIsoSequence+
    process.elePFIsoReader
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


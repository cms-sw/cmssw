import FWCore.ParameterSet.Config as cms

from TrackingTools.Configuration.TrackingTools_cff import *

process = cms.Process("electrons")

process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.GeometryPilot2_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("Configuration.StandardSequences.RawToDigi_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.EventContent.EventContent_cff")

process.source = cms.Source("PoolSource",
    debugVerbosity = cms.untracked.uint32(1),
    debugFlag = cms.untracked.bool(True),
    fileNames = cms.untracked.vstring('file:SingleElectronPt10Raw.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *', 
#        'keep recoSuperClusters*_*_*_*', 
        'keep *_iterativeCone5CaloJets_*_*', 
        'keep *_*_*_electrons', 
        'keep *HepMCProduct_*_*_*'),
    fileName = cms.untracked.string('SingleElectronPt10.root')
)

process.gsfElectronAnalysis = cms.EDAnalyzer("GsfElectronMCAnalyzer",
    #electronCollection = cms.InputTag("pixelMatchGsfElectrons"),
    electronCollection = cms.InputTag("gsfElectrons"),
    mcTruthCollection = cms.InputTag("generator"),
    #outputFile = cms.string("gsfElectronHistos_RelVal310pre3SingleElectronPt35_ecalDriven_or_trackerDriven.root"),
    #outputFile = cms.string("gsfElectronHistos_RelVal310pre3SingleElectronPt35_ecalDriven_or_trackerDriven_nopresel.root"),
    outputFile = cms.string("gsfElectronHistos_RelVal310pre3SingleElectronPt35_ecalDriven_or_trackerDriven_test.root"),
    #outputFile = cms.string("gsfElectronHistos_RelVal310pre3SingleElectronPt35_ecalDriven_or_trackerDriven_noTEC.root"),
    #outputFile = cms.string("gsfElectronHistos_RelVal310pre3SingleElectronPt35_ecalDriven_or_trackerDriven.root"),
    #outputFile = cms.string("gsfElectronHistos_RelVal310pre3SingleElectronPt35_ecalDriven_1000evts.root"),
    #outputFile = cms.string("gsfElectronHistos_RelVal310pre3SingleElectronPt35_trackerDriven_1000evts.root"),
    #outputFile = cms.string("gsfElectronHistos_RelVal310pre3SingleElectronPt35_ecalDriven_or_trackerDriven_test.root"),
    #outputFile = cms.string("gsfElectronHistos_RelVal310pre3SingleElectronPt10_ecalDriven_or_trackerDriven.root"),
    #outputFile = cms.string("gsfElectronHistos_RelVal310pre3SingleElectronPt10_ecalDriven_1000evts.root"),
    #outputFile = cms.string("gsfElectronHistos_RelVal310pre3SingleElectronPt10_trackerDriven_1000evts.root"),
    MaxPt = cms.double(100.0),
    DeltaR = cms.double(0.05),
    MaxAbsEta = cms.double(2.5),
    Etamin = cms.double(-2.5),
    Etamax = cms.double(2.5),
    Phimax = cms.double(3.2),
    Phimin = cms.double(-3.2),
    Ptmax = cms.double(100.0),
    Pmax = cms.double(300.0),
    Eopmax = cms.double(5.0),
    Eopmaxsht = cms.double(3.0),
    Detamin = cms.double(-0.005),
    Detamax = cms.double(0.005),
    Dphimin = cms.double(-0.01),
    Dphimax = cms.double(0.01),
    Dphimatchmin = cms.double(-0.2),
    Dphimatchmax = cms.double(0.2),
    Detamatchmin = cms.double(-0.05),
    Detamatchmax = cms.double(0.05),
    Fhitsmax = cms.double(20.0),
    Lhitsmax = cms.double(10.0),
    Nbinxyz = cms.int32(50),
    Nbineop2D = cms.int32(30),
    Nbinp = cms.int32(50),
    Nbineta2D = cms.int32(50),
    Nbinfhits = cms.int32(20),
    Nbinlhits = cms.int32(5),
    Nbinpteff = cms.int32(19),
    Nbinphi2D = cms.int32(32),
    Nbindetamatch2D = cms.int32(50),
    Nbineta = cms.int32(50),
    Nbinp2D = cms.int32(50),
    Nbindeta = cms.int32(100),
    Nbinpt2D = cms.int32(50),
    Nbindetamatch = cms.int32(100),
    Nbinphi = cms.int32(64),
    Nbindphimatch = cms.int32(100),
    Nbinpt = cms.int32(50),
    Nbindphimatch2D = cms.int32(50),
    Nbindphi = cms.int32(100),
    Nbineop = cms.int32(50)
)

#process.Timing = cms.Service("Timing")

#process.mylocalreco =  cms.Sequence(process.trackerlocalreco*process.calolocalreco+process.particleFlowCluster)
#process.myglobalreco = cms.Sequence(process.offlineBeamSpot+process.recopixelvertexing*process.ckftracks+process.ecalClusters+process.caloTowersRec*process.vertexreco*electronGsfTracking*process.gsfElectronSequence) 
#process.p = cms.Path(process.RawToDigi*process.mylocalreco*process.myglobalreco*process.gsfElectronAnalysis)

process.load("RecoParticleFlow.PFProducer.pfElectronTranslator_cff")
process.load("RecoEgamma.EgammaElectronProducers.gsfElectronCkfTrackCandidateMaker_cff")
process.load("RecoEgamma.EgammaElectronProducers.gsfElectronGsfFit_cff")
process.mylocalreco =  cms.Sequence(process.trackerlocalreco*process.calolocalreco)
#process.myglobalreco = cms.Sequence(process.offlineBeamSpot+process.recopixelvertexing*process.ckftracks+process.ecalClusters+process.caloTowersRec*process.vertexreco*process.pixelMatchGsfElectrons)
process.myglobalreco = cms.Sequence(process.offlineBeamSpot+process.recopixelvertexing*process.ckftracks+process.ecalClusters+process.caloTowersRec*process.vertexreco*process.particleFlowCluster)
process.myelectronseeding = cms.Sequence(process.trackerDrivenElectronSeeds*process.ecalDrivenElectronSeeds*process.electronMergedSeeds)
process.myelectrontracking = cms.Sequence(process.electronCkfTrackCandidates*process.electronGsfTracks)
process.p = cms.Path(process.RawToDigi*process.mylocalreco*process.myglobalreco*process.myelectronseeding*process.myelectrontracking*process.particleFlowReco*process.pfElectronTranslator*process.gsfElectronSequence)
#process.p = cms.Path(process.RawToDigi*process.mylocalreco*process.myglobalreco*process.myelectronseeding*process.myelectrontracking*process.particleFlowReco*process.pfElectronTranslator*process.gsfElectronSequence*process.gsfElectronAnalysis)

# to switch on only one seeding mode
#process.electronCkfTrackCandidates.src = cms.InputTag('ecalDrivenElectronSeeds')
#process.electronCkfTrackCandidates.src = cms.InputTag('trackerDrivenElectronSeeds:SeedsForGsf')

# to add pflow electrons or not
#process.gsfElectrons.addPflowElectrons = cms.bool(True)
process.gsfElectrons.addPflowElectrons = cms.bool(True)
process.gsfElectrons.applyAmbResolution = cms.bool(True)
#process.gsfElectrons.seedFromTEC = cms.bool(False)
process.gsfElectrons.applyPreselection = cms.bool(False)

process.outpath = cms.EndPath(process.out)
process.GlobalTag.globaltag = 'IDEAL_30X::All'




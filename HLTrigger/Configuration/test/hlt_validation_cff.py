#pre6
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

# setup 'standard' options
options = VarParsing.VarParsing ('standard')

## options.register('trackCollection',
##                  hltIter4Merged,
##                  VarParsing.VarParsing.multiplicity.singleton,
##                  VarParsing.VarParsing.varType.string,
##                  "choose the track collection to validate")

process = cms.Process( "iterative" )
#process.load("setup_cff")

process.HLTConfigVersion = cms.PSet(
    tableName = cms.string('/users/tropiano/2012/test_tracking/iterative/V4')
    )

#from Suva's code
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')

process.SimpleMemoryCheck=cms.Service("SimpleMemoryCheck")

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('Revision: 1.115'),
    annotation = cms.untracked.string('reco nevts:1'),
    name = cms.untracked.string('PyReleaseValidation')
)

process.options = cms.untracked.PSet(
    makeTriggerResults = cms.untracked.bool(False),
    wantSummary = cms.untracked.bool(False),
    Rethrow = cms.untracked.vstring('ProductNotFound'),
    fileMode = cms.untracked.string('FULLMERGE')
)

## need this at the end as the validation config redefines random seed with just mix
process.load("IOMC.RandomEngine.IOMC_cff")

process.load('SimGeneral.TrackingAnalysis.trackingParticles_cfi')
process.load('SimGeneral.PileupInformation.AddPileupSummary_cfi')
process.load('Validation.TrackingMCTruth.trackingTruthValidation_cfi')

process.load("Validation.RecoTrack.cuts_cff")
process.load("Validation.RecoTrack.MultiTrackValidator_cff")
process.load("Validation.Configuration.postValidation_cff")

#if(options.trackCollection = 'hltPixelTracks'):
#uncomment to do the validation on pixel tracks!!!
#scommentare la riga sotto solo per le PixelTracks
#process.TrackAssociatorByHits.SimToRecoDenominator = cms.string('reco')
    
process.load('SimTracker.TrackAssociation.TrackAssociatorByChi2_cfi')
process.load("SimTracker.TrackAssociation.TrackAssociatorByHits_cfi")
process.load('SimTracker.TrackAssociation.TrackAssociatorByPosition_cff')
process.load("SimTracker.TrackAssociation.trackingParticleRecoTrackAsssociation_cfi")

from SimTracker.TrackAssociation.TrackAssociatorByChi2_cfi import *
from SimTracker.TrackAssociation.TrackAssociatorByHits_cfi import *
process.load("Validation.RecoTrack.cutsTPEffic_cfi")
process.load("Validation.RecoTrack.cutsTPFake_cfi")
process.load('Configuration.StandardSequences.Validation_cff')
#from Suva
process.trackValidator.label=cms.VInputTag(cms.InputTag("hltIter4Merged") )
# ecco perche` di default usa questa - e solo questa - collezione!!
process.trackValidator.outputFile = 'comppixel_a.root'
process.load("DQMServices.Components.EDMtoMEConverter_cff")

# Output definition
process.output = cms.OutputModule("PoolOutputModule",
                                  splitLevel = cms.untracked.int32(0),
                                  outputCommands = process.RECOSIMEventContent.outputCommands,
                                  fileName = cms.untracked.string('out_validation.root'),
                                  dataset = cms.untracked.PSet(
    dataTier = cms.untracked.string('GEN-SIM-RECO'),
    filterName = cms.untracked.string('')
    )
                                  )

#I'm only interested in the validation stuff
process.output.outputCommands = cms.untracked.vstring('drop *',
                                                      'keep *_MEtoEDMConverter_*_*')
process.endjob_step             = cms.Path(process.endOfProcess)
process.out_step                = cms.EndPath(process.output)
#process.endjob_step = cms.Path(process.output)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
    )	

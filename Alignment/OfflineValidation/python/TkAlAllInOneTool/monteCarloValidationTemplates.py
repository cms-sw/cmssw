######################################################################
######################################################################
mcValidateTemplate="""
import FWCore.ParameterSet.Config as cms

process = cms.Process("TkVal")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('LOGFILE_McValidate_.oO[name]Oo.', 
        'cout')
)

### standard includes
process.load('Configuration.Geometry.GeometryPilot2_cff')
process.load("Configuration.StandardSequences.RawToDigi_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences..oO[magneticField]Oo._cff")
process.load("SimGeneral.MixingModule.mixNoPU_cfi")

### conditions
process.load("Alignment.OfflineValidation.GlobalTag_cff")
process.GlobalTag.globaltag = '.oO[GlobalTag]Oo.'
process.es_prefer_GlobalTag = cms.ESPrefer("PoolDBESSource", "GlobalTag")

.oO[condLoad]Oo.


### validation-specific includes
process.load("SimTracker.TrackAssociatorProducers.trackAssociatorByHits_cfi")
process.load("Validation.RecoTrack.cuts_cff")
process.load("Validation.RecoTrack.MultiTrackValidator_cff")
process.load("SimGeneral.TrackingAnalysis.trackingParticles_cfi")

### configuration MultiTrackValidator ###
process.multiTrackValidator.outputFile = '.oO[outputFile]Oo.'

process.multiTrackValidator.associators = ['trackAssociatorByHits']
process.multiTrackValidator.UseAssociators = cms.bool(True)
process.multiTrackValidator.label = ['generalTracks']

.oO[datasetDefinition]Oo.
process.source.inputCommands = cms.untracked.vstring('keep *', 'drop *_MEtoEDMConverter_*_*') # hack to get rid of the memory consumption problem in 2_2_X and beond

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(False),
    Rethrow = cms.untracked.vstring("ProductNotFound"), # make this exception fatal
    fileMode  =  cms.untracked.string('NOMERGE') # no ordering needed, but calls endRun/beginRun etc. at file boundaries
)

process.re_tracking_and_TP = cms.Sequence(process.mix*process.trackingParticles*
                                   process.siPixelRecHits*process.siStripMatchedRecHits*
                                   process.ckftracks*
                                   process.cutsRecoTracks*
                                   process.trackAssociatorByHits*
                                   process.multiTrackValidator
                                   )

process.re_tracking = cms.Sequence(process.siPixelRecHits*process.siStripMatchedRecHits*
                                   process.ckftracks*
                                   process.cutsRecoTracks*
                                   process.trackAssociatorByHits*
                                   process.multiTrackValidator
                                   )

### final path and endPath
process.p = cms.Path(process.re_tracking)
"""


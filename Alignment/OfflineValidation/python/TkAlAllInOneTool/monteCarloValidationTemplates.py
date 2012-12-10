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
process.load('Configuration/StandardSequences/GeometryPilot2_cff')
process.load("Configuration.StandardSequences.RawToDigi_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("SimGeneral.MixingModule.mixNoPU_cfi")

### conditions
process.load("Alignment.OfflineValidation.GlobalTag_cff")
process.GlobalTag.globaltag = '.oO[GlobalTag]Oo.'


.oO[dbLoad]Oo.

.oO[condLoad]Oo.

.oO[APE]Oo.

.oO[kinksAndBows]Oo.


### validation-specific includes
process.load("SimTracker.TrackAssociation.TrackAssociatorByHits_cfi")
process.load("Validation.RecoTrack.cuts_cff")
process.load("Validation.RecoTrack.MultiTrackValidator_cff")
process.load("SimGeneral.TrackingAnalysis.trackingParticles_cfi")

### configuration MultiTrackValidator ###
process.multiTrackValidator.outputFile = '.oO[outputFile]Oo.'

process.multiTrackValidator.associators = ['TrackAssociatorByHits']
process.multiTrackValidator.UseAssociators = cms.bool(True)
process.multiTrackValidator.label = ['generalTracks']

from Alignment.OfflineValidation..oO[RelValSample]Oo._cff import readFiles
from Alignment.OfflineValidation..oO[RelValSample]Oo._cff import secFiles
source = cms.Source ("PoolSource",
    fileNames = readFiles,
    secondaryFileNames = secFiles,
    inputCommands = cms.untracked.vstring('keep *', 'drop *_MEtoEDMConverter_*_*') # hack to get rid of the memory consumption problem in 2_2_X and beond
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(.oO[nEvents]Oo.)
)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(False),
    Rethrow = cms.untracked.vstring("ProductNotFound"), # make this exception fatal
    fileMode  =  cms.untracked.string('NOMERGE') # no ordering needed, but calls endRun/beginRun etc. at file boundaries
)

process.source = source

process.re_tracking_and_TP = cms.Sequence(process.mix*process.trackingParticles*
                                   process.siPixelRecHits*process.siStripMatchedRecHits*
                                   process.ckftracks*
                                   process.cutsRecoTracks*
                                   process.multiTrackValidator
                                   )

process.re_tracking = cms.Sequence(process.siPixelRecHits*process.siStripMatchedRecHits*
                                   process.ckftracks*
                                   process.cutsRecoTracks*
                                   process.multiTrackValidator
                                   )

### final path and endPath
process.p = cms.Path(process.re_tracking)
"""


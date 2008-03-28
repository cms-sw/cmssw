# The following comments couldn't be translated into the new config version:

# selection of alignables and their parameters:
# comma separated pairs of detector parts/levels as defined in AlinmentParameterSelector
# (note special meaning if the string contains "SS" or "DS" or ends with "Layers"
# followed by two digits)
# and of d.o.f. to be aligned (x,y,z,alpha,beta,gamma) in local frame:
# '0' means: deselect, '1' select. Others as 1, but might be interpreted in a special
# way in the used algorithm (e.g. 'f' means fixed for millepede)

# can add more:
import FWCore.ParameterSet.Config as cms

# misalignment scenarios
from Alignment.TrackerAlignment.Scenarios_cff import *
#  replace TrackerShortTermScenario.TPBs.scale = 10.
#  replace TrackerShortTermScenario.TPEs.scale = 10.
# algorithms
from Alignment.HIPAlignmentAlgorithm.HIPAlignmentAlgorithm_cfi import *
from Alignment.MillePedeAlignmentAlgorithm.MillePedeAlignmentAlgorithm_cfi import *
from Alignment.KalmanAlignmentAlgorithm.KalmanAlignmentAlgorithm_cfi import *
looper = cms.Looper("AlignmentProducer",
    # Save alignment corrections to DB: if true, requires configuration of PoolDBOutputService
    # See DBConfiguration.cff for an example
    saveToDB = cms.bool(False),
    doMuon = cms.untracked.bool(False),
    MisalignmentScenario = cms.PSet(
        NoMovementsScenario
    ),
    ParameterStore = cms.PSet(
        ExtendedCorrelationsConfig = cms.PSet(
            CutValue = cms.double(0.95),
            Weight = cms.double(0.5),
            MaxUpdates = cms.int32(5000)
        ),
        UseExtendedCorrelations = cms.untracked.bool(False)
    ),
    # number of selected alignables to be kept fixed
    nFixAlignables = cms.int32(0),
    # Misalignment from database: if true, requires configuration of PoolDBESSource
    # See DBConfiguration.cff for an example
    applyDbAlignment = cms.untracked.bool(False),
    monitorConfig = cms.PSet(
        monitors = cms.untracked.vstring()
    ),
    tjTkAssociationMapTag = cms.InputTag("TrackRefitter"),
    # Choose one algorithm
    algoConfig = cms.PSet(
        HIPAlignmentAlgorithm
    ),
    ParameterBuilder = cms.PSet(
        Selector = cms.PSet(
            alignParams = cms.vstring('PixelHalfBarrelLayers,111000')
        )
    ),
    # simple misalignment applied to selected alignables and selected dof
    parameterSelectorSimple = cms.string('-1'),
    randomShift = cms.double(0.0),
    doTracker = cms.untracked.bool(True),
    # misalignment scenario
    doMisalignmentScenario = cms.bool(False),
    maxLoops = cms.untracked.uint32(1),
    randomRotation = cms.double(0.0),
    # Read survey info from DB: if true, requires configuration of PoolDBESSource
    # See Alignment/SurveyAnalysis/test/readDB.cfg for an example
    useSurvey = cms.bool(False)
)



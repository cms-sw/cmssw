# The following comments couldn't be translated into the new config version:

# replace tag in MillePede
# replace tag in MillePede
import FWCore.ParameterSet.Config as cms

from Alignment.TrackerAlignment.Scenarios_cff import *
from Alignment.HIPAlignmentAlgorithm.HIPAlignmentAlgorithm_cfi import *
from Alignment.MillePedeAlignmentAlgorithm.MillePedeAlignmentAlgorithm_cfi import *
# include "Alignment/KalmanAlignmentAlgorithm/data/KalmanAlignmentAlgorithm.cfi"
# Patch for track refitter (adapted to alignment needs)
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilderWithoutRefit_cfi import *
from RecoTracker.TrackProducer.RefitterWithMaterial_cff import *
# Database output service
# Required if AlignmentProducer.saveToDB = true
from CondCore.DBCommon.CondDBSetup_cfi import *
looper = cms.Looper("AlignmentProducer",
    saveToDB = cms.bool(False),
    doMuon = cms.untracked.bool(False),
    MisalignmentScenario = cms.PSet(
        TrackerCSA07Scenario
    ),
    ParameterStore = cms.PSet(
        ExtendedCorrelationsConfig = cms.PSet(
            CutValue = cms.double(0.95),
            Weight = cms.double(0.5),
            MaxUpdates = cms.int32(5000)
        ),
        UseExtendedCorrelations = cms.untracked.bool(False)
    ),
    nFixAlignables = cms.int32(0), ## number of selected alignables to be fixed

    applyDbAlignment = cms.untracked.bool(True),
    monitorConfig = cms.PSet(
        monitors = cms.untracked.vstring('AlignmentMonitorGeneric'),
        AlignmentMonitorGeneric = cms.untracked.PSet(
            outfile = cms.string('histograms.root'),
            collectorActive = cms.bool(False),
            collectorPath = cms.string('./'),
            outpath = cms.string('./'), ## to be replaced accordingly

            collectorNJobs = cms.int32(0)
        )
    ),
    tjTkAssociationMapTag = cms.InputTag("TrackRefitter"),
    algoConfig = cms.PSet( ## to be replaced


    ),
    ParameterBuilder = cms.PSet(
        Selector = cms.PSet( ## to be replaced by Millepede

            alignParams = cms.vstring('TOBDSRods,111111', 'TOBSSRodsLayers15,100111', 'TIBDSDets,111111', 'TIBSSDets,100111')
        )
    ),
    # do not apply simple misalignment
    parameterSelectorSimple = cms.string('-1'),
    randomShift = cms.double(0.0),
    doTracker = cms.untracked.bool(True),
    doMisalignmentScenario = cms.bool(False),
    maxLoops = cms.untracked.uint32(1), ## to be replaced by HIP

    randomRotation = cms.double(0.0),
    useSurvey = cms.bool(False)
)

AlignDBSetup = cms.PSet(
    CondDBSetup,
    timetype = cms.string('runnumber'),
    connect = cms.string('oracle://cms_orcoff_int2r/CMS_COND_ALIGNMENT')
)
PoolDBESSource = cms.ESSource("PoolDBESSource",
    AlignDBSetup,
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('TrackerAlignmentRcd'),
        tag = cms.string('TrackerCSA07Scenario160')
    ), cms.PSet(
        record = cms.string('TrackerAlignmentErrorRcd'),
        tag = cms.string('TrackerCSA07ScenarioErrors160')
    ))
)

PoolDBOutputService = cms.Service("PoolDBOutputService",
    AlignDBSetup,
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('TrackerAlignmentRcd'),
        tag = cms.string('TrackerCSA07HIPAlignments')
    ), cms.PSet(
        record = cms.string('TrackerAlignmentErrorRcd'),
        tag = cms.string('TrackerCSA07HIPAlignmentErrors')
    ))
)

TrackRefitter.src = 'ALCARECOTkAlZMuMu'
TrackRefitter.TTRHBuilder = 'WithoutRefit'
TrackRefitter.TrajectoryInEvent = True
ttrhbwor.Matcher = 'StandardMatcher' ## matching for strip stereo!



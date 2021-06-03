import FWCore.ParameterSet.Config as cms

# misalignment scenarios
from Alignment.TrackerAlignment.Scenarios_cff import *

# algorithms
from Alignment.HIPAlignmentAlgorithm.HIPAlignmentAlgorithm_cfi import *
from Alignment.MillePedeAlignmentAlgorithm.MillePedeAlignmentAlgorithm_cfi import *

# parameters
from Alignment.CommonAlignmentAlgorithm.AlignmentParameterStore_cfi import *

looper = cms.Looper("AlignmentProducer",
                    AlignmentParameterStore, # configuration of AlignmentParameterStore
                    maxLoops = cms.untracked.uint32(1),
                    doTracker = cms.untracked.bool(True),
                    doMuon = cms.untracked.bool(False),
                    useExtras = cms.untracked.bool(False),
                    # Read survey info from DB: true requires configuration of PoolDBESSource
                    # See Alignment/SurveyAnalysis/test/readDB.cfg for an example
                    useSurvey = cms.bool(False),
                    
                    # (Mis-)alignment including surface deformations from database
                    # true requires configuration of PoolDBESSource
                    applyDbAlignment = cms.untracked.bool(False),
                                        
                    # Checks the IOV of the alignment to be applied. Only has an effect
                    # if applyDbAlignment is True as well. If set to True, the alignment
                    # record to be applied is expected to have a validity from 1 to INF
                    checkDbAlignmentValidity = cms.untracked.bool(True),

                    # misalignment scenario
                    MisalignmentScenario = cms.PSet(NoMovementsScenario), # why not by reference?
                    doMisalignmentScenario = cms.bool(False),
                    # simple misalignment of selected alignables and selected dof (deprecated!)
                    randomShift = cms.double(0.0),
                    randomRotation = cms.double(0.0),
                    parameterSelectorSimple = cms.string('-1'),
                    
                    # selection of alignables and their parameters
                    # see twiki: SWGuideAlignmentAlgorithms
                    ParameterBuilder = cms.PSet(parameterTypes = cms.vstring('Selector,RigidBody'),
                                                Selector = cms.PSet(alignParams = cms.vstring('PixelHalfBarrelLayers,111000'))
                                                ),
                    # number of selected alignables to be kept fixed (deprecated!)
                    nFixAlignables = cms.int32(0), # i.e. removed from selection above...

                    # event input
                    tjTkAssociationMapTag = cms.InputTag("TrackRefitter"),
                    beamSpotTag           = cms.InputTag("offlineBeamSpot"),
                    hitPrescaleMapTag     = cms.InputTag(""), # not used if empty
                    # run input
                    tkLasBeamTag          = cms.InputTag(""), # not used if empty
                    
                    # Choose one algorithm with configuration, HIP is default
                    algoConfig = cms.PSet(HIPAlignmentAlgorithm), # why not by reference?
                    # Some algorithms support integrated calibrations, which to use is defined
                    # by the string 'calibrationName' in the PSet of each calibration.
                    calibrations = cms.VPSet(),
                    # choose monitors (default is none)
                    monitorConfig = cms.PSet(monitors = cms.untracked.vstring()),

                    # VPSet that allows splitting of alignment parameters into various
                    # run ranges. The default is a run range independent alignment
                    RunRangeSelection = cms.VPSet(
                      #cms.PSet(RunRanges = cms.vstring('-1','140401','143488')
                      #         selector = cms.vstring('TrackerTPBHalfBarrel,001000',
                      #                                'TrackerTPEHalfDisk,111000')
                      #)
                    ),


                    # Save alignment to DB: true requires configuration of PoolDBOutputService
                    saveToDB = cms.bool(False),             # save alignment?
                    saveApeToDB = cms.bool(False),          # save APE?
                    saveDeformationsToDB = cms.bool(False), # save surface deformations (bows, etc.)?

                    # update alignables if triggered by corresponding input IOV boundary
                    enableAlignableUpdates = cms.bool(False),
                    )
DTGeometryAlignmentProducer = cms.ESProducer("DTGeometryESModule",
    appendToDataLabel = cms.string('idealForAlignmentProducerBase'),
    applyAlignment = cms.bool(False), ## to be abondoned (?)
    alignmentsLabel = cms.string(''),
    fromDDD = cms.bool(True)
)
CSCGeometryAlignmentProducer = cms.ESProducer("CSCGeometryESModule",
    appendToDataLabel = cms.string('idealForAlignmentProducerBase'),
    debugV = cms.untracked.bool(False),
    useGangedStripsInME1a = cms.bool(False),
    alignmentsLabel = cms.string(''),
    useOnlyWiresInME1a = cms.bool(False),
    useRealWireGeometry = cms.bool(True),
    useCentreTIOffsets = cms.bool(False),
    applyAlignment = cms.bool(False), ## GF: to be abandoned
    fromDDD = cms.bool(True),
    fromDD4hep = cms.bool(False)
) 
GEMGeometryAlignmentProducer = cms.ESProducer("GEMGeometryESModule",
    appendToDataLabel = cms.string('idealForAlignmentProducerBase'),
    applyAlignment = cms.bool(False),
    alignmentsLabel = cms.string(''),
    fromDDD = cms.bool(True),
    fromDD4Hep = cms.bool(False)
)

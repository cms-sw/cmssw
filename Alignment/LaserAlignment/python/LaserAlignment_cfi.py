
import FWCore.ParameterSet.Config as cms

# main configuration for LaserAlignment reconstruction module

LaserAlignment = cms.EDFilter( "LaserAlignment",

    # create a plain ROOT file containing the collected profile histograms?
    SaveHistograms = cms.untracked.bool( False ),
    ROOTFileName = cms.untracked.string('LaserAlignment.histos.root'),
    ROOTFileCompression = cms.untracked.int32( 1 ),
                               
    # list of digi producers
    DigiProducersList = cms.VPSet(cms.PSet(
        DigiLabel = cms.string('\0'),
        DigiProducer = cms.string('siStripDigis')
    )),
    
    # enable the zero (empty profile) filter in the LASProfileJudge, so profiles without signal are rejected.
    # might want to disable this for simulated data with typically low signal level on the last disks
    EnableJudgeZeroFilter = cms.untracked.bool(True),

    # if this is set to true, the geometry update is applied to the ideal geometry, not to the input geometry.
    # should be set false only for geometry comparison purposes (see TkLasCMSSW Twiki for more details)
    UpdateFromIdealGeometry = cms.untracked.bool( True ),

    # whether to create an sqlite file with a TrackerAlignmentRcd + error
    SaveToDbase = cms.untracked.bool(True),

    # do pedestal subtraction for raw digis. DISABLE THIS for simulated or zero suppressed data
    SubtractPedestals = cms.untracked.bool(True),

)




import FWCore.ParameterSet.Config as cms

# main configuration for LaserAlignment reconstruction module

LaserAlignment = cms.EDFilter( "LaserAlignment",


    ### USER OPTIONS:

    # create a plain ROOT file containing the collected profile histograms?
    SaveHistograms = cms.untracked.bool( False ),
    ROOTFileName = cms.untracked.string( 'LAS.histos.root' ),
    ROOTFileCompression = cms.untracked.int32( 1 ),
                               
    # list of digi producers
    DigiProducersList = cms.VPSet(
      cms.PSet(
        DigiLabel = cms.string( '\0' ),
        DigiProducer = cms.string( 'siStripDigis' ),
        DigiType = cms.string( 'Raw' )
      )
    ),
    
    # enable the zero (empty profile) filter in the LASProfileJudge, so profiles without signal are rejected.
    # might want to disable this for simulated data with typically low signal level on the last disks
    EnableJudgeZeroFilter = cms.untracked.bool( True ),

    # if this is set to True, the geometry update is reversely applied to the input geometry (not to IDEAL).
    # should be set false only for geometry comparison purposes with MC (see TkLasCMSSW Twiki for more details)
    UpdateFromInputGeometry = cms.untracked.bool( False ),

    # whether to create an sqlite file with a TrackerAlignmentRcd + error
    SaveToDbase = cms.untracked.bool( True ),

    # do pedestal subtraction for raw digis. DISABLE THIS for simulated or zero suppressed data
    SubtractPedestals = cms.untracked.bool( False ),

    # if true run the MINUIT based AT algorithm rather than the analytical one
    RunMinuitAlignmentTubeAlgorithm = cms.untracked.bool( False ),



    ### TESTING OPTIONS (EXPERTS ONLY):

    # override LASPeakFinder results and set hit strip numbers to nominal                               
    ForceFitterToNominalStrips = cms.untracked.bool( False ) # NOT YET IN USE

                               
)



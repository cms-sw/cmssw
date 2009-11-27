# Auto generated configuration file
# using: 
# Revision: 1.149 
# Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: promptCollisionReco -s RAW2DIGI,L1Reco,RECO,DQM,ALCA:SiStripCalZeroBias --datatier RECO --eventcontent RECO --conditions CRAFT09_R_V4::All --scenario pp --no_exec --data --magField AutoFromDBCurrent -n 100
import FWCore.ParameterSet.Config as cms

process = cms.Process('RECO')

# import of standard configurations
process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load('Configuration/StandardSequences/GeometryExtended_cff')
process.load('Configuration/StandardSequences/MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration/StandardSequences/RawToDigi_Data_cff')
process.load('Configuration/StandardSequences/L1Reco_cff')
process.load('Configuration/StandardSequences/Reconstruction_cff')
process.load('DQMOffline/Configuration/DQMOffline_cff')
process.load('Configuration/StandardSequences/AlCaRecoStreams_cff')
process.load('Configuration/StandardSequences/EndOfProcess_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load('Configuration/EventContent/EventContent_cff')

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    annotation = cms.untracked.string('promptCollisionReco nevts:100'),
    name = cms.untracked.string('PyReleaseValidation')
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5)
)
process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound'),
    wantSummary = cms.untracked.bool(True) 
)
# Input source
process.source = cms.Source("PoolSource",
    skipEvents = cms.untracked.uint32(278), 
#    fileNames = cms.untracked.vstring('/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/122/270/747C6137-F0D7-DE11-BE6C-001D09F242EF.root')
    fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/cms/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/122/314/CC89C4BC-60D8-DE11-B365-0030487D0D3A.root')
)

# Output definition
process.FEVT = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    outputCommands = process.RECOEventContent.outputCommands,
    fileName = cms.untracked.string('promptReco_RAW2DIGI_L1Reco_RECO_DQM_ALCA.root'),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('RECO'),
        filterName = cms.untracked.string('')
    )
)

# Combined AlCaReco output
process.ALCARECOStreamCombined = cms.OutputModule("PoolOutputModule",
    outputCommands = process.ALCARECOEventContent.outputCommands,
    fileName = cms.untracked.string('ALCACombined.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('StreamALCACombined'),
        dataTier = cms.untracked.string('ALCARECO')
    )
)
process.ALCARECOStreamCombined.outputCommands.extend(cms.untracked.vstring('drop *_MEtoEDMConverter_*_*'))

# Other statements
process.GlobalTag.globaltag = 'GR09_P_V6::All'



#####################################################################################################
####
####  Top level replaces for handling strange scenarios of early collisions
####

## TRACKING:
process.globalPixelLessSeeds.RegionFactoryPSet.RegionPSet.ptMin = 0.5
process.pixelLessCkfTrajectoryFilter = process.ckfBaseTrajectoryFilter.clone(
    ComponentName = 'pixelLessCkfTrajectoryFilter',
    filterPset = process.ckfBaseTrajectoryFilter.filterPset.clone(minPt = 0.5)
    )
process.pixelLessCkfTrajectoryBuilder = process.GroupedCkfTrajectoryBuilder.clone(
    ComponentName = 'pixelLessCkfTrajectoryBuilder',
    trajectoryFilterName = 'pixelLessCkfTrajectoryFilter',
    )
process.ckfTrackCandidatesPixelLess.TrajectoryBuilder = 'pixelLessCkfTrajectoryBuilder'
process.globalPixelLessSeeds.RegionFactoryPSet.RegionPSet.originHalfLength = 40
process.globalPixelLessSeeds.ClusterCheckPSet.MaxNumberOfCosmicClusters = 5000
## Skip events with HV off
process.fourthPLSeeds.ClusterCheckPSet.MaxNumberOfCosmicClusters = 20000
process.fifthSeeds.ClusterCheckPSet.MaxNumberOfCosmicClusters = 5000
## Seeding: increase the region
process.fifthSeeds.RegionFactoryPSet.RegionPSet.originHalfLength = 100
process.fifthSeeds.RegionFactoryPSet.RegionPSet.originRadius     =  20
## Seeding: add TOB3 to the list, allow unmatched hits
process.fifthlayerpairs.TOB.useSimpleRphiHitsCleaner = cms.bool(False)
process.fifthlayerpairs.TOB.rphiRecHits = cms.InputTag("fifthStripRecHits","rphiRecHitUnmatched")
process.fifthlayerpairs.TOB.stereoRecHits = cms.InputTag("fifthStripRecHits","stereoRecHitUnmatched")
process.fifthlayerpairs.layerList += [ 'TOB1+TOB3', 'TOB2+TOB3' ]
#, 'TOB3+TOB4' ]
## Pattern recognition: lower the cut on the number of hits
process.fifthCkfTrajectoryFilter.filterPset.minimumNumberOfHits = 5
process.fifthCkfTrajectoryFilter.filterPset.maxLostHits = 3
process.fifthCkfTrajectoryFilter.filterPset.maxConsecLostHits = 1
process.fifthCkfInOutTrajectoryFilter.filterPset.minimumNumberOfHits = 3
process.fifthCkfInOutTrajectoryFilter.filterPset.maxLostHits = 3
process.fifthCkfInOutTrajectoryFilter.filterPset.maxConsecLostHits = 1
process.fifthCkfTrajectoryBuilder.minNrOfHitsForRebuild = 3
## Pattern recognition: enlarge a lot the search window, as the true momentum is very small while the tracking assumes p=5 GeV if B=0
process.Chi2MeasurementEstimator.MaxChi2 = 200
process.Chi2MeasurementEstimator.nSigma  = 3
## Fitter-smoother: lower the cut on the number of hits
process.fifthRKTrajectorySmoother.minHits = 4
process.fifthRKTrajectoryFitter.minHits = 4
process.fifthFittingSmootherWithOutlierRejection.MinNumberOfHits = 5
## Fitter-smoother: loosen outlier rejection
process.fifthFittingSmootherWithOutlierRejection.BreakTrajWith2ConsecutiveMissing = False
process.fifthFittingSmootherWithOutlierRejection.EstimateCut = 1000
## Quality filter
process.tobtecStepLoose.minNumberLayers = 3
process.tobtecStepLoose.minNumber3DLayers = 0
process.tobtecStepLoose.maxNumberLostLayers = 4
process.tobtecStepLoose.dz_par1 = cms.vdouble(100.5, 4.0)
process.tobtecStepLoose.dz_par2 = cms.vdouble(100.5, 4.0)
process.tobtecStepLoose.d0_par1 = cms.vdouble(100.5, 4.0)
process.tobtecStepLoose.d0_par2 = cms.vdouble(100.5, 4.0)
process.tobtecStepLoose.chi2n_par = cms.double(10.0)
process.tobtecStepLoose.keepAllTracks = False
process.tobtecStepTight = process.tobtecStepLoose.clone(
    keepAllTracks = True,
    qualityBit = cms.string('tight'),
    src = cms.InputTag("tobtecStepLoose"),
    minNumberLayers = 5,
    minNumber3DLayers = 0
    )
process.tobtecStep = process.tobtecStepLoose.clone(
    keepAllTracks = True,
    qualityBit = cms.string('highPurity'),
    src = cms.InputTag("tobtecStepTight"),
    minNumberLayers = 4,
    minNumber3DLayers = 2,
    )

## PV temporary fixes
process.offlinePrimaryVertices.PVSelParameters.maxDistanceToBeam = 10
process.offlinePrimaryVertices.TkFilterParameters.maxNormalizedChi2 = 500
process.offlinePrimaryVertices.TkFilterParameters.minSiliconHits = 5
process.offlinePrimaryVertices.TkFilterParameters.maxD0Significance = 100
process.offlinePrimaryVertices.TkFilterParameters.minPixelHits = -1
process.offlinePrimaryVertices.TkClusParameters.zSeparation = 10

## ECAL temporary fixes
process.load('RecoLocalCalo.EcalRecProducers.ecalFixedAlphaBetaFitUncalibRecHit_cfi')
process.ecalLocalRecoSequence.replace(process.ecalGlobalUncalibRecHit,process.ecalFixedAlphaBetaFitUncalibRecHit)
process.ecalRecHit.EBuncalibRecHitCollection = 'ecalFixedAlphaBetaFitUncalibRecHit:EcalUncalibRecHitsEB'
process.ecalRecHit.EEuncalibRecHitCollection = 'ecalFixedAlphaBetaFitUncalibRecHit:EcalUncalibRecHitsEE'
process.ecalRecHit.ChannelStatusToBeExcluded = [ 1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 78, 142 ]

## HCAL temporary fixes
process.hfreco.firstSample  = 3
process.hfreco.samplesToAdd = 4

###
###  end of top level replacements
###
###############################################################################################

# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.RawToDigi)
process.L1Reco_step = cms.Path(process.L1Reco)
process.reconstruction_step = cms.Path(process.reconstruction_withPixellessTk)
process.dqmoffline_step = cms.Path(process.DQMOffline)
process.pathALCARECOHcalCalHOCosmics = cms.Path(process.seqALCARECOHcalCalHOCosmics)
process.pathALCARECOMuAlStandAloneCosmics = cms.Path(process.seqALCARECOMuAlStandAloneCosmics*process.ALCARECOMuAlStandAloneCosmicsDQM)
process.pathALCARECOTkAlZMuMu = cms.Path(process.seqALCARECOTkAlZMuMu*process.ALCARECOTkAlZMuMuDQM)
process.pathALCARECOTkAlCosmicsCTF0T = cms.Path(process.seqALCARECOTkAlCosmicsCTF0T*process.ALCARECOTkAlCosmicsCTF0TDQM)
process.pathALCARECOMuAlBeamHalo = cms.Path(process.seqALCARECOMuAlBeamHalo*process.ALCARECOMuAlBeamHaloDQM)
process.pathALCARECOTkAlCosmicsRS0THLT = cms.Path(process.seqALCARECOTkAlCosmicsRS0THLT*process.ALCARECOTkAlCosmicsRS0TDQM)
process.pathALCARECOTkAlCosmicsCTF = cms.Path(process.seqALCARECOTkAlCosmicsCTF*process.ALCARECOTkAlCosmicsCTFDQM)
process.pathALCARECOHcalCalIsoTrk = cms.Path(process.seqALCARECOHcalCalIsoTrk*process.ALCARECOHcalCalIsoTrackDQM)
process.pathALCARECOHcalCalHO = cms.Path(process.seqALCARECOHcalCalHO*process.ALCARECOHcalCalHODQM)
process.pathALCARECOTkAlCosmicsCTFHLT = cms.Path(process.seqALCARECOTkAlCosmicsCTFHLT*process.ALCARECOTkAlCosmicsCTFDQM)
process.pathALCARECOTkAlCosmicsRS0T = cms.Path(process.seqALCARECOTkAlCosmicsRS0T*process.ALCARECOTkAlCosmicsRS0TDQM)
process.pathALCARECOTkAlCosmicsCosmicTFHLT = cms.Path(process.seqALCARECOTkAlCosmicsCosmicTFHLT*process.ALCARECOTkAlCosmicsCosmicTFDQM)
process.pathALCARECOHcalCalMinBias = cms.Path(process.seqALCARECOHcalCalMinBias*process.ALCARECOHcalCalPhisymDQM)
process.pathALCARECOTkAlMuonIsolated = cms.Path(process.seqALCARECOTkAlMuonIsolated*process.ALCARECOTkAlMuonIsolatedDQM)
process.pathALCARECOTkAlUpsilonMuMu = cms.Path(process.seqALCARECOTkAlUpsilonMuMu*process.ALCARECOTkAlUpsilonMuMuDQM)
process.pathALCARECOHcalCalDijets = cms.Path(process.seqALCARECOHcalCalDijets*process.ALCARECOHcalCalDiJetsDQM)
process.pathALCARECOMuAlZMuMu = cms.Path(process.seqALCARECOMuAlZMuMu*process.ALCARECOMuAlZMuMuDQM)
process.pathALCARECOEcalCalPi0Calib = cms.Path(process.seqALCARECOEcalCalPi0Calib*process.ALCARECOEcalCalPi0CalibDQM)
process.pathALCARECOTkAlBeamHalo = cms.Path(process.seqALCARECOTkAlBeamHalo*process.ALCARECOTkAlBeamHaloDQM)
process.pathALCARECOSiPixelLorentzAngle = cms.Path(process.seqALCARECOSiPixelLorentzAngle)
process.pathALCARECOTkAlCosmicsCosmicTF0T = cms.Path(process.seqALCARECOTkAlCosmicsCosmicTF0T*process.ALCARECOTkAlCosmicsCosmicTF0TDQM)
process.pathALCARECOEcalCalElectron = cms.Path(process.seqALCARECOEcalCalElectron*process.ALCARECOEcalCalElectronCalibDQM)
process.pathALCARECOTkAlCosmicsCTF0THLT = cms.Path(process.seqALCARECOTkAlCosmicsCTF0THLT*process.ALCARECOTkAlCosmicsCTF0TDQM)
process.pathALCARECOMuAlCalIsolatedMu = cms.Path(process.seqALCARECOMuAlCalIsolatedMu*process.ALCARECOMuAlCalIsolatedMuDQM*process.ALCARECODTCalibrationDQM)
process.pathALCARECOSiStripCalZeroBias = cms.Path(process.seqALCARECOSiStripCalZeroBias*process.ALCARECOSiStripCalZeroBiasDQM)
process.pathALCARECOTkAlCosmicsRSHLT = cms.Path(process.seqALCARECOTkAlCosmicsRSHLT*process.ALCARECOTkAlCosmicsRSDQM)
process.pathALCARECOEcalCalEtaCalib = cms.Path(process.seqALCARECOEcalCalEtaCalib*process.ALCARECOEcalCalEtaCalibDQM)
process.pathALCARECOSiStripCalMinBias = cms.Path(process.seqALCARECOSiStripCalMinBias)
process.pathALCARECODQM = cms.Path(process.MEtoEDMConverter)
process.pathALCARECOTkAlLAS = cms.Path(process.seqALCARECOTkAlLAS*process.ALCARECOTkAlLASDQM)
process.pathALCARECOTkAlMinBias = cms.Path(process.seqALCARECOTkAlMinBias*process.ALCARECOTkAlMinBiasDQM)
process.pathALCARECOTkAlCosmicsRS = cms.Path(process.seqALCARECOTkAlCosmicsRS*process.ALCARECOTkAlCosmicsRSDQM)
process.pathALCARECORpcCalHLT = cms.Path(process.seqALCARECORpcCalHLT)
process.pathALCARECOHcalCalGammaJet = cms.Path(process.seqALCARECOHcalCalGammaJet)
process.pathALCARECOMuAlBeamHaloOverlaps = cms.Path(process.seqALCARECOMuAlBeamHaloOverlaps*process.ALCARECOMuAlBeamHaloOverlapsDQM)
process.pathALCARECOTkAlCosmicsCosmicTF0THLT = cms.Path(process.seqALCARECOTkAlCosmicsCosmicTF0THLT*process.ALCARECOTkAlCosmicsCosmicTF0TDQM)
process.pathALCARECOHcalCalNoise = cms.Path(process.seqALCARECOHcalCalNoise)
process.pathALCARECOMuAlOverlaps = cms.Path(process.seqALCARECOMuAlOverlaps*process.ALCARECOMuAlOverlapsDQM)
process.pathALCARECOTkAlCosmicsCosmicTF = cms.Path(process.seqALCARECOTkAlCosmicsCosmicTF*process.ALCARECOTkAlCosmicsCosmicTFDQM)
process.pathALCARECOEcalCalPhiSym = cms.Path(process.seqALCARECOEcalCalPhiSym*process.ALCARECOEcalCalPhisymDQM)
process.pathALCARECOMuAlGlobalCosmics = cms.Path(process.seqALCARECOMuAlGlobalCosmics*process.ALCARECOMuAlGlobalCosmicsDQM)
process.pathALCARECOTkAlJpsiMuMu = cms.Path(process.seqALCARECOTkAlJpsiMuMu*process.ALCARECOTkAlJpsiMuMuDQM)
process.endjob_step = cms.Path(process.endOfProcess)
process.out_step = cms.EndPath(process.FEVT)
process.ALCARECOStreamCombinedOutPath = cms.EndPath(process.ALCARECOStreamCombined)

# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,process.L1Reco_step,process.reconstruction_step,process.dqmoffline_step,process.pathALCARECOSiStripCalZeroBias,process.pathALCARECOTkAlMinBias,process.pathALCARECOTkAlMuonIsolated,process.pathALCARECOMuAlCalIsolatedMu,process.pathALCARECOMuAlOverlaps,process.pathALCARECOHcalCalIsoTrk,process.pathALCARECOHcalCalDijets,process.endjob_step,process.out_step,process.ALCARECOStreamCombinedOutPath)

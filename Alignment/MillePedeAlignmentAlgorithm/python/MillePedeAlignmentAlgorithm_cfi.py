# MillePedeAlignmentAlgorithm
# ---------------------------

import FWCore.ParameterSet.Config as cms

from Alignment.ReferenceTrajectories.TrajectoryFactories_cff import *

MillePedeAlignmentAlgorithm = cms.PSet(
    algoName = cms.string('MillePedeAlignmentAlgorithm'),
    mode = cms.untracked.string('full'), ## possible modes: full, mille, pede, pedeSteer, pedeRun, pedeRead
    fileDir = cms.untracked.string(''),

    # Where mille writes (and pede reads) derivatives, labels etc.
    binaryFile = cms.string('milleBinary.dat'),
    # Resulting and initial parameters, absolute (original) positions, result etc.
    treeFile = cms.string('treeFile.root'),
    # Must be empty if mille runs, otherwise for merging (pede) jobs should be parallel with each
    # other. Then 'treeFile' is merged result and 'binaryFile' should be empty.
    mergeBinaryFiles = cms.vstring(),
    mergeTreeFiles = cms.vstring(),

    monitorFile = cms.untracked.string('millePedeMonitor.root'), ## if empty: no monitoring...

    pedeSteerer = cms.PSet(
        fileDir = cms.untracked.string(''),
        steerFile = cms.string('pedeSteer'), ## beginning of steering file names
        # If MillePedeAlignmentAlgorithm.mode causes pede to run (e.g. 'full', 'pede' etc.),
        # the pede command line is constructed as:
        #    'pedeCommand' 'steerFile'Master.txt 
        # (and - if pedeDump is not empty - extended by: > 'pedeDump')
        # (MillePedeAlignmentAlgorithm.theDir is taken into account...)
        pedeCommand = cms.untracked.string('/afs/cern.ch/user/f/flucke/cms/pede/versWebEndMay2007/pede'),

        parameterSign = cms.untracked.int32(1), ## old pede versions (before May '07) need a sign flip
        pedeDump = cms.untracked.string('pede.dump'),
        method = cms.string('sparseGMRES 6  0.8'), ## "inversion  6  0.8" 
        options = cms.vstring('entries 50', # min. number of measurements (parameters with less will be skipped)
            # "chisqcut  20.0  4.5", # simple chi^2 cut for outliers OR ...
            # "outlierdownweighting 3", "dwfractioncut 0.1" #, # ... 3x Huber down weighting OR...
            'outlierdownweighting 4', 
            'dwfractioncut 0.2', 
            'bandwidth 6'), # for method sparseGMRES for preconditioning

        # Special selection of parameters to fix, use as reference coordinate system etc.
        # ------------------------------------------------------------------------------
        # All this is determined from what is given as 
        # AlignmentProducer.ParameterBuilder.Selector, cf. Twiki page SWGuideMillepedeIIAlgorithm.
        Presigmas = cms.VPSet()
    ),

    pedeReader = cms.PSet(
        readFile = cms.string('millepede.res'),
        # directory of 'readFile', if empty:
	# take from pedeSteerer (inheriting from MillePedeAlignmentAlgorithm)
        fileDir = cms.untracked.string('')
    ),

    readPedeInput = cms.bool(False), ## if true, following used to read in pede result as input
    pedeReaderInput = cms.PSet(
        fileDir = cms.untracked.string('./'),
        readFile = cms.string('millepede.res')
    ),

    TrajectoryFactory = cms.PSet(
        ReferenceTrajectoryFactory # from TrajectoryFactories
	# BzeroReferenceTrajectoryFactory
	# TwoBodyDecayReferenceTrajectoryFactory
    ),
    minNumHits = cms.int32(5), ## minimum number of hits (with alignable parameters)
    max2Dcorrelation = cms.double(0.05), ## if correlation >5% 2D measurements get diagonalized
    useTrackTsos = cms.bool(False) ## Tsos from track or from reference trajectory for global derivatives
)


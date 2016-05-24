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

    # PSet that allows to configure the pede labeler, i.e. select the actual
    # labeler plugin to use and parameters for the selected plugin
    pedeLabeler = cms.PSet(
      #plugin = cms.string('MomentumDependentPedeLabeler')
      #parameterInstances = cms.VPSet(
      #  cms.PSet(momentumRanges = cms.vstring('0.0:50.0','50.0:10000.0'),
      #           selector = cms.vstring('ExtrasBeamSpot,1111'))
      #  )
    ),
    
    pedeSteerer = cms.PSet(
        fileDir = cms.untracked.string(''),
        steerFile = cms.string('pedeSteer'), ## beginning of steering file names
        steerFileDebug = cms.untracked.bool(False),
        # If MillePedeAlignmentAlgorithm.mode causes pede to run (e.g. 'full', 'pede' etc.),
        # the pede command line is constructed as:
        #    'pedeCommand' 'steerFile'Master.txt 
        # (and - if pedeDump is not empty - extended by: > 'pedeDump')
        # (MillePedeAlignmentAlgorithm.theDir is taken into account...)
        pedeCommand = cms.untracked.string('pede'),

        parameterSign = cms.untracked.int32(1), ## old pede versions (before May '07) need a sign flip
        pedeDump = cms.untracked.string('pede.dump'),
        method = cms.string('sparseMINRES 6  0.8'), ## "inversion  6  0.8" 
        options = cms.vstring('entries 50', # min. number of measurements (parameters with less will be skipped)
            # 'regularisation 1.0 0.01', # regularisation with default pre-sigma 0.01
            # "chisqcut  20.0  4.5", # simple chi^2 cut for outliers AND/OR ...
            # "outlierdownweighting 3", "dwfractioncut 0.1" #, # ... 3x Huber down weighting OR...
            'outlierdownweighting 5', 'dwfractioncut 0.2'),

        # Special selection of parameters to fix, use as reference coordinate system etc.
        # ------------------------------------------------------------------------------
        # All this is determined from what is given as 
        # AlignmentProducer.ParameterBuilder.Selector, cf. Twiki page SWGuideMillepedeIIAlgorithm.
        Presigmas = cms.VPSet(),
        minHieraConstrCoeff = cms.double(1.e-7), # min abs value of coeff. in hierarchy constr.
        minHieraParPerConstr = cms.uint32(2), # ignore hierarchy constraints with less params
        constrPrecision = cms.uint32(0), # use default precision for writing constraints to text file

        # specify additional steering files
        additionalSteerFiles = cms.vstring(), # obsolete - can be given as entries in 'options'
        
        # Parameter vector for the systematic geometry deformations
        # Empty vector -> constraints are NOT applied (default)
        constraints = cms.VPSet()
    ),

    pedeReader = cms.PSet(
        readFile = cms.string('millepede.res'),
        # directory of 'readFile', if empty:
        # take from pedeSteerer (inheriting from MillePedeAlignmentAlgorithm)
        fileDir = cms.untracked.string('')
    ),

    # Array of PSet's like 'pedeReader' above to be applied before running mille,
    # i.e. for iterative running of Millepede without going via DB constants
    # (note: if 'fileDir' empty, the one from 'pedeSteerer' will be used...): 
    pedeReaderInputs = cms.VPSet(),

    TrajectoryFactory = BrokenLinesTrajectoryFactory,  # from TrajectoryFactories
	# BrokenLinesBzeroTrajectoryFactory
	# TwoBodyDecayReferenceTrajectoryFactory, # for this overwrite MaterialEffects for BL
    minNumHits = cms.uint32(7), ## minimum number of hits (with alignable parameters)
    max2Dcorrelation = cms.double(0.05), ## if correlation >5% 2D measurements in TID/TEC get diagonalized
    doubleBinary = cms.bool(False), ## create binary files with doubles instead of floats (GBL only)

	# Parameters for PXB survey steering
    surveyPixelBarrel = cms.PSet(
			doSurvey = cms.bool(False),
			infile = cms.FileInPath("Alignment/SurveyAnalysis/data/BPix_Survey_info_raw.txt"),
			doOutputOnStdout = cms.bool(False),
			# Settings for toy survey - produces a file with toy survey data according to given parameters
			doToySurvey = cms.bool(False),
			toySurveyFile = cms.untracked.string('toySurveyInfo.txt'),
			toySurveySeed = cms.uint32(12),
			toySurveyParameters = cms.VPSet(
					# Position of photo in local frame (unit: pixels in photo)
					cms.PSet(name = cms.string('a0'),    mean = cms.double(1800.), sigma = cms.double(150.)),
					cms.PSet(name = cms.string('a1'),    mean = cms.double(2600.), sigma = cms.double(200.)),
					# Scale of photo (unit: pixels per cm)
					cms.PSet(name = cms.string('scale'), mean = cms.double(1150.), sigma = cms.double(50.)),
					# Rotation of photo in local frame (unit: rads)
					cms.PSet(name = cms.string('phi'),   mean = cms.double(0.), sigma = cms.double(0.0025)),
					# Smearing of measurements in u and v coordinate (unit: pixels in photo)
					cms.PSet(name = cms.string('u'),     mean = cms.double(0.), sigma = cms.double(0.175)),
					cms.PSet(name = cms.string('v'),     mean = cms.double(0.), sigma = cms.double(0.175))
					)
	)
)


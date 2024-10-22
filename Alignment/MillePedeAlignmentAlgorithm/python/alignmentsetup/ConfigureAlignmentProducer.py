import FWCore.ParameterSet.Config as cms

def setConfiguration(process, collection, mode, monitorFile, binaryFile,
                     primaryWidth = 0.0, cosmicsZeroTesla = False):

    #############
    ## general ##
    #############
    process.load("Alignment.CommonAlignmentProducer.AlignmentProducerAsAnalyzer_cff")

    # Start geometry from db
    process.AlignmentProducer.applyDbAlignment         = True
    process.AlignmentProducer.checkDbAlignmentValidity = False

    # What tracks are used to construct the reference trajectories?
    process.AlignmentProducer.tjTkAssociationMapTag = "FinalTrackRefitter"

    # enable proper handling of multi-IOV input
    process.AlignmentProducer.enableAlignableUpdates = True

    # Configure the algorithm
    process.AlignmentProducer.algoConfig = cms.PSet(
        process.MillePedeAlignmentAlgorithm)
    process.AlignmentProducer.algoConfig.mode              = mode
    process.AlignmentProducer.algoConfig.mergeBinaryFiles  = cms.vstring()
    process.AlignmentProducer.algoConfig.skipGlobalPositionRcdCheck = True

    # default pede options:
    process.AlignmentProducer.algoConfig.pedeSteerer.method = "sparseMINRES-QLP 3  0.8"
    process.AlignmentProducer.algoConfig.minNumHits = 8
    process.AlignmentProducer.algoConfig.pedeSteerer.options = [
        "entries 50 10 2",
        "outlierdownweighting 3",
        "dwfractioncut 0.1",
        "compress",
        "threads 10 10",
        "matiter 1",
        "printcounts 2",
        "chisqcut  30.  6.",
        "bandwidth 6 1",
        "monitorresiduals",
    ]

    if mode == "mille":
        process.AlignmentProducer.algoConfig.binaryFile   = binaryFile
        process.AlignmentProducer.algoConfig.monitorFile  = monitorFile
    elif "pede" in mode:
        process.AlignmentProducer.algoConfig.binaryFile   = ''
        process.AlignmentProducer.algoConfig.monitorFile  = 'millePedeMonitor_merge.root'
        process.AlignmentProducer.algoConfig.treeFile     = 'treeFile_merge.root'


    ########################
    ## Tracktype specific ##
    ########################

    if collection == "ALCARECOTkAlZMuMu" or collection == "ALCARECOTkAlZMuMuHI" or collection == "ALCARECOTkAlZMuMuPA" or collection == "ALCARECOTkAlDiMuon":
        process.AlignmentProducer.algoConfig.TrajectoryFactory = cms.PSet(
             process.TwoBodyDecayTrajectoryFactory
        )
        process.AlignmentProducer.algoConfig.TrajectoryFactory.ParticleProperties.PrimaryMass = 91.1061
        process.AlignmentProducer.algoConfig.TrajectoryFactory.ParticleProperties.PrimaryWidth = 1.7678
        process.AlignmentProducer.algoConfig.TrajectoryFactory.MaterialEffects = "LocalGBL"
        # to account for multiple scattering in these layers
        process.AlignmentProducer.algoConfig.TrajectoryFactory.UseInvalidHits = True
    elif collection == "ALCARECOTkAlUpsilonMuMu":
        process.AlignmentProducer.algoConfig.TrajectoryFactory = cms.PSet(
             process.TwoBodyDecayTrajectoryFactory
        )
        process.AlignmentProducer.algoConfig.TrajectoryFactory.ParticleProperties.PrimaryMass = 9.4502
        process.AlignmentProducer.algoConfig.TrajectoryFactory.ParticleProperties.PrimaryWidth = 0.0644
        process.AlignmentProducer.algoConfig.TrajectoryFactory.MaterialEffects = "LocalGBL"
        # to account for multiple scattering in these layers
        process.AlignmentProducer.algoConfig.TrajectoryFactory.UseInvalidHits = True
    elif collection == "ALCARECOTkAlJpsiMuMu":
        process.AlignmentProducer.algoConfig.TrajectoryFactory = cms.PSet(
             process.TwoBodyDecayTrajectoryFactory
        )
        process.AlignmentProducer.algoConfig.TrajectoryFactory.ParticleProperties.PrimaryMass = 3.0969
        process.AlignmentProducer.algoConfig.TrajectoryFactory.ParticleProperties.PrimaryWidth = 0.03
        process.AlignmentProducer.algoConfig.TrajectoryFactory.MaterialEffects = "LocalGBL"
        # to account for multiple scattering in these layers
        process.AlignmentProducer.algoConfig.TrajectoryFactory.UseInvalidHits = True
    elif collection == "ALCARECOTkAlCosmicsCTF0T" and cosmicsZeroTesla:
        process.AlignmentProducer.algoConfig.TrajectoryFactory = cms.PSet(
            process.BrokenLinesBzeroTrajectoryFactory
        )
        process.AlignmentProducer.algoConfig.TrajectoryFactory.MaterialEffects = "LocalGBL"
        process.AlignmentProducer.algoConfig.TrajectoryFactory.MomentumEstimate = 5.0
    else:
        process.AlignmentProducer.algoConfig.TrajectoryFactory = cms.PSet(
            process.BrokenLinesTrajectoryFactory
        )
        process.AlignmentProducer.algoConfig.TrajectoryFactory.MaterialEffects = "LocalGBL"


    ##################
    ## primaryWidth ##
    ##################

    if primaryWidth > 0.0:
        process.AlignmentProducer.algoConfig.TrajectoryFactory.ParticleProperties.PrimaryWidth = primaryWidth


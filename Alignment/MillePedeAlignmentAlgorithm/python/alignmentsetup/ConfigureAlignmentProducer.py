import FWCore.ParameterSet.Config as cms

def setConfiguration(process, collection, mode, monitorFile, binaryFile, primaryWidth = 0.0):

    #############
    ## general ##
    #############

    # Start geometry from db
    process.AlignmentProducer.applyDbAlignment         = True
    process.AlignmentProducer.checkDbAlignmentValidity = False

    # What tracks are used to construct the reference trajectories?
    process.AlignmentProducer.tjTkAssociationMapTag = "FinalTrackRefitter"

    # Configure the algorithm
    process.AlignmentProducer.algoConfig = cms.PSet(
        process.MillePedeAlignmentAlgorithm)
    process.AlignmentProducer.algoConfig.mode              = mode
    process.AlignmentProducer.algoConfig.mergeBinaryFiles  = cms.vstring()

    if mode == "mille":
        process.AlignmentProducer.algoConfig.binaryFile   = binaryFile
        process.AlignmentProducer.algoConfig.monitorFile  = monitorFile
    elif "pede" in mode:
        process.AlignmentProducer.algoConfig.binaryFile   = ''
        process.AlignmentProducer.algoConfig.monitorFile  = 'millePedeMonitor_merge.root'
        process.AlignmentProducer.algoConfig.treeFile     = 'treeFile_merge.root'
        process.AlignmentProducer.algoConfig.pedeSteerer.pedeCommand = "pede"


    ########################
    ## Tracktype specific ##
    ########################

    if collection is "ALCARECOTkAlZMuMu":
        process.AlignmentProducer.algoConfig.TrajectoryFactory = cms.PSet(
             process.TwoBodyDecayTrajectoryFactory
        )
        process.AlignmentProducer.algoConfig.TrajectoryFactory.ParticleProperties.PrimaryMass = 90.8745
        process.AlignmentProducer.algoConfig.TrajectoryFactory.ParticleProperties.PrimaryWidth = 1.8770
        process.AlignmentProducer.algoConfig.TrajectoryFactory.MaterialEffects = "LocalGBL"
        # to account for multiple scattering in these layers
        process.AlignmentProducer.algoConfig.TrajectoryFactory.UseInvalidHits = True
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


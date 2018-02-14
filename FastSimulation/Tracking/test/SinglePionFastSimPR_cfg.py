import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(==NUMEVT==)
)

process.load("FWCore.MessageService.MessageLogger_cfi")

# import of standard configurations
process.load("IOMC.RandomEngine.IOMC_cff")
process.load('FastSimulation.PileUpProducer.PileUpSimulator10TeV_cfi')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load('Configuration/StandardSequences/Generator_cff')
process.load('FastSimulation/Configuration/FamosSequences_cff')
process.load('FastSimulation/Configuration/HLT_cff')
process.load('Configuration.StandardSequences.L1TriggerDefaultMenu_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedParameters_cfi')
process.load('FastSimulation/Configuration/CommonInputs_cff')
process.load('FastSimulation/Configuration/EventContent_cff')

# DiPions in energy bins

process.source = cms.Source("EmptySource")

# Other statements
process.famosPileUp.PileUpSimulator = process.PileUpSimulatorBlock.PileUpSimulator
process.famosPileUp.PileUpSimulator.averageNumber = 0
process.famosSimHits.SimulateCalorimetry = False
process.famosSimHits.SimulateTracking = True
process.famosSimHits.ActivateDecays.comEnergy = 10000
process.simulation = cms.Sequence(process.simulationWithFamos)
process.HLTEndSequence = cms.Sequence(process.reconstructionWithFamos)

# set correct vertex smearing
process.Early10TeVCollisionVtxSmearingParameters.type = cms.string("BetaFunc")
process.famosSimHits.VertexGenerator = process.Early10TeVCollisionVtxSmearingParameters
process.famosPileUp.VertexGenerator = process.Early10TeVCollisionVtxSmearingParameters
process.GlobalTag.globaltag = 'IDEAL_31X::All'

# Include the RandomNumberGeneratorService definition
process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.generator.initialSeed= ==seed1==
#process.RandomNumberGeneratorService.theSource.initialSeed= 1414

# DiPions in energy bins
process.generator = cms.EDProducer(
    "FlatRandomPtGunProducer",
    firstRun = cms.untracked.uint32(1),
    PGunParameters = cms.PSet(
        PartID = cms.vint32(211),
        MinPt = cms.double(==MINPT==.0),
        MaxPt = cms.double(==MAXPT==.0),
        MinEta = cms.double(-2.8),
        MaxEta = cms.double(+2.8),
        MinPhi = cms.double(-3.14159265359), ## it must be in radians
        MaxPhi = cms.double(3.14159265359),
    ),
    AddAntiParticle = cms.bool(False), # back-to-back particles
    Verbosity = cms.untracked.int32(0) ## for printouts, set it to 1 (or greater)
)    
process.ProductionFilterSequence = cms.Sequence(process.generator)
# this example configuration offers some minimum 
# annotation, to help users get through; please
# don't hesitate to read through the comments
# use MessageLogger to redirect/suppress multiple
# service messages coming from the system
#
# in this config below, we use the replace option to make
# the logger let out messages of severity ERROR (INFO level
# will be suppressed), and we want to limit the number to 10
#

process.load("RecoLocalTracker.Configuration.RecoLocalTracker_cff")
process.load("FastSimulation.TrackingRecHitProducer.SiClusterTranslator_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("RecoTracker.Configuration.RecoTracker_cff")

#CPEs
process.load("FastSimulation.TrackingRecHitProducer.FastPixelCPE_cfi")
process.load("FastSimulation.TrackingRecHitProducer.FastStripCPE_cfi")

#First Step
process.load("RecoTracker.IterativeTracking.FirstStep_cff")
process.newClusters.pixelClusters = cms.InputTag('siClusterTranslator')
process.newClusters.stripClusters = cms.InputTag('siClusterTranslator')
process.newPixelRecHits.CPE = cms.string('FastPixelCPE')
process.newStripRecHits.StripCPE = cms.string('FastStripCPE')
process.newMeasurementTracker.StripCPE = cms.string('FastStripCPE')
process.newMeasurementTracker.PixelCPE = cms.string('FastPixelCPE')

#Second Step
process.load("RecoTracker.IterativeTracking.SecStep_cff")
process.secPixelRecHits.CPE = cms.string('FastPixelCPE')
process.secStripRecHits.StripCPE = cms.string('FastStripCPE')
process.secMeasurementTracker.StripCPE = cms.string('FastStripCPE')
process.secMeasurementTracker.PixelCPE = cms.string('FastPixelCPE')

#Third Step
process.load("RecoTracker.IterativeTracking.ThStep_cff")
process.thPixelRecHits.CPE = cms.string('FastPixelCPE')
process.thStripRecHits.StripCPE = cms.string('FastStripCPE')
process.thMeasurementTracker.StripCPE = cms.string('FastStripCPE')
process.thMeasurementTracker.PixelCPE = cms.string('FastPixelCPE')

#Fourth Step
process.load("RecoTracker.IterativeTracking.PixelLessStep_cff")
process.fourthPixelRecHits.CPE = cms.string('FastPixelCPE')
process.fourthStripRecHits.StripCPE = cms.string('FastStripCPE')
process.fourthMeasurementTracker.StripCPE = cms.string('FastStripCPE')
process.fourthMeasurementTracker.PixelCPE = cms.string('FastPixelCPE')

#Fifth Step
process.load("RecoTracker.IterativeTracking.TobTecStep_cff")
process.fifthPixelRecHits.CPE = cms.string('FastPixelCPE')
process.fifthStripRecHits.StripCPE = cms.string('FastStripCPE')
process.fifthMeasurementTracker.StripCPE = cms.string('FastStripCPE')
process.fifthMeasurementTracker.PixelCPE = cms.string('FastPixelCPE')

#Strips
process.load("RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi")
process.siStripMatchedRecHits.StripCPE = cms.string('FastStripCPE')
process.siStripMatchedRecHits.ClusterProducer = cms.string('siClusterTranslator')

#Pixels
process.load("RecoLocalTracker.SiPixelRecHits.PixelCPEESProducers_cff")
process.load("RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi")
process.siPixelRecHits.src = cms.InputTag('siClusterTranslator')
process.siPixelRecHits.CPE = cms.string('FastPixelCPE')
process.load("RecoTracker.TkSeedGenerator.GlobalSeedsFromTripletsWithVertices_cff")
process.globalSeedsFromTripletsWithVertices.TTRHBuilder = cms.string("FastPixelCPE")

#Transient Rec Hits
process.load("RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi")
process.ttrhbwr.StripCPE = cms.string('FastStripCPE')
process.ttrhbwr.PixelCPE = cms.string('FastPixelCPE')
process.load("RecoTracker.TransientTrackingRecHit.TTRHBuilderWithTemplate_cfi")
process.TTRHBuilderAngleAndTemplate.StripCPE = cms.string('FastStripCPE')
process.TTRHBuilderAngleAndTemplate.PixelCPE = cms.string('FastPixelCPE')
process.load("RecoTracker.TkSeedingLayers.TTRHBuilderWithoutAngle4PixelPairs_cfi")
process.myTTRHBuilderWithoutAngle4PixelPairs.PixelCPE = cms.string("FastPixelCPE")
process.load("RecoTracker.TkSeedingLayers.TTRHBuilderWithoutAngle4PixelTriplets_cfi")
process.myTTRHBuilderWithoutAngle4PixelTriplets.PixelCPE = cms.string("FastPixelCPE")
process.load("RecoTracker.TkSeedingLayers.TTRHBuilderWithoutAngle4MixedPairs_cfi")
process.myTTRHBuilderWithoutAngle4MixedPairs.PixelCPE = cms.string("FastPixelCPE")
process.load("RecoTracker.TkSeedingLayers.TTRHBuilderWithoutAngle4MixedTriplets_cfi")
process.myTTRHBuilderWithoutAngle4MixedTriplets.PixelCPE = cms.string("FastPixelCPE")

#Tracks
process.load("RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi")
process.MeasurementTracker.stripClusterProducer = cms.string('siClusterTranslator')
process.MeasurementTracker.pixelClusterProducer = cms.string('siClusterTranslator')
process.MeasurementTracker.StripCPE = cms.string('FastStripCPE')
process.MeasurementTracker.PixelCPE = cms.string('FastPixelCPE')

# Event output
process.load("Configuration.EventContent.EventContent_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")

process.MessageLogger = cms.Service("MessageLogger",
    reco = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG')
    ),
   destinations = cms.untracked.vstring('reco')
)

##from Kevin
process.zeroStepHighPurity = cms.EDProducer("QualityFilter",
                                           TrackQuality = cms.string('highPurity'),
                                           recTracks = cms.InputTag("zeroStepTracksWithQuality")
                                           )
process.firstStepHighPurity = cms.EDProducer("QualityFilter",
                                           TrackQuality = cms.string('highPurity'),
                                           recTracks = cms.InputTag("preMergingFirstStepTracksWithQuality")
                                           )
process.fifthStepHighPurity = cms.EDProducer("QualityFilter",
                                           TrackQuality = cms.string('highPurity'),
                                           recTracks = cms.InputTag("tobtecStep")
                                           )
process.fevt = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string("fevt.root"),
    outputCommands = cms.untracked.vstring(
      'drop *',
      ###---these are the collection used in input to the "general tracks"
      'keep *_zeroStepHighPurity*_*_*',
      'keep *_firstStepHighPurity*_*_*',
      'keep *_firstfilter*_*_*',
      'keep *_secfilter*_*_*',
      'keep *_thfilter*_*_*',
      'keep *_fourthfilter*_*_*',
      'keep *_thfilter*_*_*',
      'keep *_fifthStepHighPurity*_*_*',
      # zero step 
      'keep *_zeroStep*_*_*',
      # step one
      'keep *_preMergingFirstStepTracksWithQuality*_*_*',
      # first(merged 0+1)  iterative step
      'keep *_firstStep*_*_*',
      # second step high quality
      'keep *_secStep_*_*',
      #third step  high quality
      'keep *_thStep_*_*',
      #fourth step high quality
      'keep *_pixellessStep*_*_*',
      #fifth step high quality
      'keep *_tobtecStep*_*_*',
      # merge of secStep+thStep 
      'keep *_merge2nd3rdTracks*_*_*',
      # merge of merge2nd3rd+pixelless
      'keep *_iterTracks*_*_*',
      # merge of pixellessStep+tobtecStep 
      'keep *_merge4th5thTracks*_*_*',
      #merge of firstStepTracksWithQuality+iterTracks
      "keep *_generalTracks_*_*",      
      'keep *_*Seed*_*_*',
      'keep *_sec*_*_*',
      'keep *_th*_*_*',
      'keep *_fou*_*_*',
      'keep *_fifth*_*_*',
      'keep *_newTrackCandidateMaker_*_*',
      "keep SimTracks_*_*_*",
      "keep SimVertexs_*_*_*",
      "keep edmHepMCProduct_*_*_*"
      )
)


# Produce Tracks and Clusters
process.generation_step = cms.Path(cms.SequencePlaceholder("randomEngineStateProducer")+process.GeneInfo)
process.reconstruction = cms.Path(process.famosWithTrackerHits+process.siClusterTranslator+process.siPixelRecHits+
                                  process.siStripMatchedRecHits+process.recopixelvertexing+
                                  process.iterTracking+
                                  process.zeroStepHighPurity+
                                  process.firstStepHighPurity+
                                  process.fifthStepHighPurity
                                  )


process.out_step = cms.EndPath(process.fevt)

# Schedule definition
process.schedule = cms.Schedule(process.generation_step)
process.schedule.extend([process.reconstruction,process.out_step])

# special treatment in case of production filter sequence  
for path in process.paths: 
    getattr(process,path)._seq = process.ProductionFilterSequence*getattr(process,path)._seq




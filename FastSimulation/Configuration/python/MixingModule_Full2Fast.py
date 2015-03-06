#####################################
# customisation functions that allow to convert a FullSim PU cfg into a FastSim one
# main functions: prepareGenMixing and prepareDigiRecoMixing
# author: Lukas Vanelderen
# date:   Jan 21 2015
#####################################

import FWCore.ParameterSet.Config as cms

def get_VertexGeneratorPSet_PileUpProducer(process):

    # container for vertex parameters
    vertexParameters = cms.PSet()

    # find the standard vertex generator
    if not hasattr(process,"VtxSmeared"):
        "WARNING: no vtx smearing applied (ok for steps other than SIM)"
        return vertexParameters
    vertexGenerator = process.VtxSmeared

    # check the type of the standard vertex generator
    vertexGeneratorType = vertexGenerator.type_().replace("EvtVtxGenerator","")
    vtxGenMap = {"Betafunc":"BetaFunc","Flat":"Flat","Gauss":"Gaussian"}
    if not vertexGeneratorType in vtxGenMap.keys():
        raise Error("WARNING: given vertex generator type for vertex smearing is not supported")
    vertexParameters.type = cms.string(vtxGenMap[vertexGeneratorType])
    
    # set vertex generator parameters in PileUpProducer
    vertexGeneratorParameterNames = vertexGenerator.parameterNames_()
    for name in vertexGeneratorParameterNames:
        exec("vertexParameters.{0} = {1}".format(name,getattr(vertexGenerator,name).dumpPython()))

    return vertexParameters

def get_PileUpSimulatorPSet_PileUpProducer(_input):

    # container for PileUpSimulator parameters
    PileUpSimulator = cms.PSet()

    # Extract the type of pu distribution
    _type = "none"
    if hasattr(_input,"type"):
        _type = _input.type

    if _type == "poisson":
        if not hasattr(_input.nbPileupEvents,"averageNumber"):
            print "  ERROR while reading PU distribution for FastSim PileUpProducer:"
            print "  when process.mix.input.type is set to 'poisson', process.mix.input.nbPileupEvents.averageNumber must be specified."
            raise
        PileUpSimulator.averageNumber = _input.nbPileupEvents.averageNumber
        PileUpSimulator.usePoisson = cms.bool(True)

    elif _type == "probFunction":
        if not hasattr(_input.nbPileupEvents,"probFunctionVariable") or not hasattr(_input.nbPileupEvents,"probValue"):
            print "  ERROR while reading PU distribution for FastSim PileUpProducer:"
            print "  when process.mix.input.type is set to 'probFunction', process.mix.nbPileupEvents.probFunctionVariable and process.mix.nbPileupEvents.probValue must be specified"
            raise
        PileUpSimulator.usePoisson = cms.bool(False)
        PileUpSimulator.probFunctionVariable = _input.nbPileupEvents.probFunctionVariable
        PileUpSimulator.probValue = _input.nbPileupEvents.probValue

    elif _type != "none":
        print "  ERROR while reading PU distribution for FastSim PileUpProducer:"
        print "  value {0} for process.mix.input.type not supported by FastSim GEN-level PU mixing".format(_type)
        raise

    # minbias files
    from FastSimulation.PileUpProducer.PileUpFiles_cff import puFileNames
    PileUpSimulator.fileNames = puFileNames.fileNames
    
    # a purely technical, but required, setting
    PileUpSimulator.inputFile = cms.untracked.string('PileUpInputFile.txt')

    return PileUpSimulator


def prepareGenMixing(process):
    
    # prepare digitizers and mixObjects for Gen-mixing
    process = prepareDigiRecoMixing(process)

    # OOT PU not supported for Gen-mixing: disable it
    process.mix.maxBunch = cms.int32(0)
    process.mix.minBunch = cms.int32(0)
    
    # set the bunch spacing
    # bunch spacing matters for calorimeter calibration
    # setting the bunch spacing here, will have actually no effect,
    # but leads to consistency with the bunch spacing as hard coded in
    # FastSimulation/PileUpProducer/plugins/PileUpProducer.cc
    # where it is propagated to the pileUpInfo, from which calorimeter calibration reads the bunch spacing
    process.mix.bunchspace = 450

    # define the PileUpProducer module
    process.famosPileUp = cms.EDProducer(
        "PileUpProducer",
        PileUpSimulator = cms.PSet(),
        VertexGenerator = cms.PSet()
        )

    # get the pu vertex distribution
    process.famosPileUp.VertexGenerator = get_VertexGeneratorPSet_PileUpProducer(process)


    # get pu distribution from MixingModule
    process.famosPileUp.PileUpSimulator = get_PileUpSimulatorPSet_PileUpProducer(process.mix.input)     

    # MixingModule only used for digitization, no need for input
    del process.mix.input

    # Insert the PileUpProducer in the simulation sequence
    pos = process.simulationSequence.index(process.famosSimHits)
    process.simulationSequence.insert(pos,process.famosPileUp)

    # No track mixing when Gen-mixing
    del process.mix.digitizers.tracker
    del process.mix.mixObjects.mixRecoTracks
    del process.generalTracks
    process.generalTracks = process.generalTracksBeforeMixing.clone()
    process.iterTracking.replace(process.generalTracksBeforeMixing,process.generalTracks)
    del process.generalTracksBeforeMixing

    # Use generalTracks where DIGI-RECO mixing requires preMixTracks
    process.generalConversionTrackProducer.TrackProducer = cms.string('generalTracks')
    # it's not nice but gen-mixing will be depricated anyhow
    process.trackerDrivenElectronSeedsTmp.TkColList = cms.VInputTag(cms.InputTag("generalTracks"))
    process.trackerDrivenElectronSeeds.oldTrackCollection = cms.InputTag('generalTracks')

    # PileUp info must be read from PileUpProducer, rather than from MixingModule
    process.addPileupInfo.PileupMixingLabel = cms.InputTag("famosPileUp")
    
    return process

def prepareDigiRecoMixing(process):

    # switch to FastSim digitizers
    import FastSimulation.Configuration.digitizers_cfi
    process.mix.digitizers = FastSimulation.Configuration.digitizers_cfi.digitizersFull2Fast(process.mix.digitizers)

    # switch to FastSim mixObjects
    import FastSimulation.Configuration.mixObjects_cfi
    process.mix.mixObjects = FastSimulation.Configuration.mixObjects_cfi.theMixObjects

    # fastsim does not simulate castor
    # fastsim does not digitize pixel and strip hits
    for element in ["simCastorDigis","simSiPixelDigis","simSiStripDigis"]:
        if hasattr(process,element):
            delattr(process,element)
    
    # import the FastSim specific EDAliases for collections from MixingModule
    process.generalTracks = FastSimulation.Configuration.digitizers_cfi.generalTracks

    # change the subject of the alias in case of pre-mixing
    if hasattr(process,"mixData"):
        process.generalTracks = cms.EDAlias(
            mixData = process.generalTracks.mix
            )

    # get rid of some FullSim specific psets that work confusing when dumping FastSim cfgs 
    # (this is optional)
    del process.theDigitizers
    del process.theDigitizersValid    
    del process.trackingParticles
    del process.stripDigitizer
    del process.SiStripSimBlock
    del process.castorDigitizer
    del process.pixelDigitizer
    del process.ecalDigitizer
    
    # get rid of FullSim specific services that work confusing when dumping FastSim cfgs
    # (this is optional)
    del process.siStripGainSimESProducer

    return process

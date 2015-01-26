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
    if not vertexGeneratorType in ["Betafunc","Flat","Gauss"]:
        raise Error("WARNING: given vertex generator type for vertex smearing is not supported")
    if vertexGeneratorType == "Gauss":
        vertexGeneratorType == "Gaussian"
    vertexParameters.type = cms.string(vertexGeneratorType)
    
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
    from FastSimulation.PileUpProducer.PileUpFiles_cff import fileNames_13TeV
    PileUpSimulator.fileNames = fileNames_13TeV
    
    # a purely technical, but required, setting
    PileUpSimulator.inputFile = cms.untracked.string('PileUpInputFile.txt')

    return PileUpSimulator


def prepareGenMixing(process):
    
    # prepare digitizers and mixObjects for Gen-mixing
    process = prepareDigiRecoMixing(process)

    # for reasons of simplicity track mixing is not switched off,
    # although it has no effect in case of Gen-mixing

    # OOT PU not supported for Gen-mixing: disable it
    process.mix.maxbunch = cms.int32(0)
    process.mix.minbunch = cms.int32(0)
    
    # set the bunch spacing
    # bunch spacing matters for calorimeter calibration
    # setting the bunch spacing here, will have actually no effect,
    # but leads to consistency with the bunch spacing as hard coded in
    # FastSimulation/PileUpProducer/plugins/PileUpProducer.cc
    # where it is propagated to the pileUpInfo, from which calorimeter calibration reads the bunch spacing
    process.mix.bunchspace = 450

    # MixingModule only used for digitization, no need for input
    _input_temp = process.mix.input
    if hasattr(process.mix,"input"):
        del process.mix.input
    
    # No track mixing when Gen-mixing
    del process.mix.digitizers.tracker
    del process.mix.mixObjects.mixRecoTracks
    del process.generalTracks
    process.generalTracks = process.generalTracksBeforeMixing.clone()
    process.lastTrackingSteps.replace(process.generalTracksBeforeMixing,process.generalTracks)
    del process.generalTracksBeforeMixing

    # Use generalTracks where DIGI-RECO mixing requires preMixTracks
    process.generalConversionTrackProducer.TrackProducer = cms.string('generalTracks')
    process.trackerDrivenElectronSeeds.TkColList = cms.VInputTag(cms.InputTag("generalTracks"))
    
    # Add the gen-level PileUpProducer to the process
    process.famosPileUp = cms.EDProducer(
        "PileUpProducer",
        PileUpSimulator = get_PileUpSimulatorPSet_PileUpProducer(_input_temp),
        VertexGenerator = get_VertexGeneratorPSet_PileUpProducer(process)
        )
    
    # Insert the PileUpProducer in the simulation sequence
    pos = process.simulationSequence.index(process.famosSimHits)
    process.simulationSequence.insert(pos,process.famosPileUp)

    # PileUp info must be read from PileUpProducer, rather than from MixingModule
    process.addPileupInfo.PileupMixingLabel = cms.InputTag("famosPileUp")
    
    return process

def prepareDigiRecoMixing(process):

    # switch to FastSim digitizers
    if hasattr(process,"theDigitizersValid"):
        del process.theDigitizersValid
    from FastSimulation.Configuration.digitizers_cfi import theDigitizersValid
    process.mix.digitizers = theDigitizersValid

    # switch to FastSim mixObjects
    if hasattr(process,"theMixObjects"):
        del process.theMixObjects
    from FastSimulation.Configuration.mixObjects_cfi import theMixObjects
    process.mix.mixObjects = theMixObjects

    # get rid of FullSim specific EDAliases for collections from MixingModule
    del process.simCastorDigis
    del process.simSiPixelDigis
    del process.simSiStripDigis

    # import the FastSim specific EDAliases for collections from MixingModule
    from FastSimulation.Configuration.digitizers_cfi import generalTracks
    process.generalTracks = generalTracks

    return process

import FWCore.ParameterSet.Config as cms

allLayer1Electrons = cms.EDProducer("PATElectronProducer",
    addElectronIDRobust = cms.bool(True), ## switch on/off the adding of additional electron id info, from the "robust cuts-based" producer            

    electronLRFile = cms.string('PhysicsTools/PatUtils/data/ElectronLRDistros.root'),
    # MC matching configurables
    addGenMatch = cms.bool(True),
    # resolution configurables
    addResolutions = cms.bool(True),
    egammaHcalIsoSource = cms.InputTag("layer0EgammaHOETower"),
    # likelihood ratio configurables
    addLRValues = cms.bool(True),
    electronResoFile = cms.string('PhysicsTools/PatUtils/data/Resolutions_electron.root'),
    addCalIsolation = cms.bool(True), ## switch on/off the calorimeter isolation calculations

    towerSource = cms.InputTag("towerMaker"), ## towers to be used for the calorimeter isolation

    egammaTkIsoSource = cms.InputTag("layer0EgammaElectronTkIsolation"),
    electronIDSource = cms.InputTag("electronId"), ## label of the electron ID info source

    # General configurables
    electronSource = cms.InputTag("allLayer0Electrons"),
    # electron ID configurables
    addElectronID = cms.bool(True),
    useNNResolutions = cms.bool(False), ## use the neural network approach?

    electronIDRobustSource = cms.InputTag("electronIdRobust"), ## label for the robust cuts-based producer (should be just a clone of previous)

    #the following tags refer to the isolation calculated by the egamma isolation tool
    #the interested user can modify the parameters of the isolation that gets put in the Electron object,
    #either by modifying the parameters of the egamma modules directly (see PATElectronIsolation.cff)
    #or by adding another copy of the producer to the path and changing these tags
    addEgammaIso = cms.bool(True),
    egammaTkNumIsoSource = cms.InputTag("layer0EgammaElectronTkNumIsolation"),
    egammaEcalIsoSource = cms.InputTag("layer0EgammaEcalRelIsolation"),
    tracksSource = cms.InputTag("ctfWithMaterialTracks"), ## tracks to be used for the tracker isolation

    # input root file for the resolution functions
    # isolation configurables
    # these following four tags refer to the isolation calculated by PAT code
    addTrkIsolation = cms.bool(True),
    genParticleMatch = cms.InputTag("electronMatch") ## Association between electrons and generator particles

)



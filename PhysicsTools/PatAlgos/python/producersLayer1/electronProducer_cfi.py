import FWCore.ParameterSet.Config as cms

allLayer1Electrons = cms.EDProducer("PATElectronProducer",
    addElectronIDRobust = cms.bool(True),
    electronLRFile = cms.string('PhysicsTools/PatUtils/data/ElectronLRDistros.root'),
    addGenMatch = cms.bool(True),
    addResolutions = cms.bool(True),
    electronResoFile = cms.string('PhysicsTools/PatUtils/data/Resolutions_electron.root'),
    addLRValues = cms.bool(True),
    isoDeposits = cms.PSet(

    ),
    electronSource = cms.InputTag("allLayer0Electrons"),
    addElectronID = cms.bool(True),
    useNNResolutions = cms.bool(False),
    electronIDRobustSource = cms.InputTag("electronIdRobust"),
    isolation = cms.PSet(
        hcal = cms.PSet(
            src = cms.InputTag("layer0ElectronIsolations","egammaTowerIsolation")
        ),
        tracker = cms.PSet(
            src = cms.InputTag("layer0ElectronIsolations","egammaElectronTkIsolation")
        ),
        user = cms.VPSet(cms.PSet(
            src = cms.InputTag("layer0ElectronIsolations","egammaElectronTkNumIsolation")
        ), 
            cms.PSet(
                src = cms.InputTag("layer0ElectronIsolations","egammaEcalRelIsolation")
            ), 
            cms.PSet(
                src = cms.InputTag("layer0ElectronIsolations","egammaHOETower")
            )),
        ecal = cms.PSet(
            src = cms.InputTag("layer0ElectronIsolations","egammaEcalIsolation")
        )
    ),
    electronIDSource = cms.InputTag("electronId"),
    tracksSource = cms.InputTag("generalTracks"),
    genParticleMatch = cms.InputTag("electronMatch")
)



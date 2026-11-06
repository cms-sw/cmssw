import FWCore.ParameterSet.Config as cms

l1tNNCaloTauEmulator = cms.EDProducer("L1NNCaloTauEmulator",
    CB_CE_split = cms.double(1.55),
    CNNmodel_CB_path = cms.string('L1Trigger/L1CaloTrigger/data/Phase2_NNCaloTaus/v22/CNNmodel_CB.pb'),
    CNNmodel_CE_path = cms.string('L1Trigger/L1CaloTrigger/data/Phase2_NNCaloTaus/v22/CNNmodel_CE.pb'),
    DEBUG = cms.bool(False),
    DNNcalib_CB_path = cms.string('L1Trigger/L1CaloTrigger/data/Phase2_NNCaloTaus/v22/DNNcalib_CB.pb'),
    DNNcalib_CE_path = cms.string('L1Trigger/L1CaloTrigger/data/Phase2_NNCaloTaus/v22/DNNcalib_CE.pb'),
    DNNident_CB_path = cms.string('L1Trigger/L1CaloTrigger/data/Phase2_NNCaloTaus/v22/DNNident_CB.pb'),
    DNNident_CE_path = cms.string('L1Trigger/L1CaloTrigger/data/Phase2_NNCaloTaus/v22/DNNident_CE.pb'),
    EcalEtMinForClustering = cms.double(0),
    EtMinForSeeding = cms.double(2.5),
    EtaRestriction = cms.double(2.4),
    FeatScaler_CE_path = cms.string('L1Trigger/L1CaloTrigger/data/Phase2_NNCaloTaus/Cl3dFeatScaler_CE.json'),
    HcalEtMinForClustering = cms.double(0),
    HgcalClusters = cms.InputTag("l1tHGCalBackEndLayer2Producer","HGCalBackendLayer2Processor3DClustering"),
    IdWp90_CB = cms.double(0.706),
    IdWp90_CE = cms.double(0.5711),
    IdWp95_CB = cms.double(0.3432),
    IdWp95_CE = cms.double(0.2742),
    IdWp99_CB = cms.double(0.0337),
    IdWp99_CE = cms.double(0.0394),
    PuidThr = cms.double(-0.1),
    VsPuId = cms.PSet(
        isPUFilter = cms.bool(True),
        method = cms.string('BDT'),
        preselection = cms.string(''),
        variables = cms.VPSet(
            cms.PSet(
                name = cms.string('eMax'),
                value = cms.string('eMax()')
            ),
            cms.PSet(
                name = cms.string('eMaxOverE'),
                value = cms.string('eMax()/energy()')
            ),
            cms.PSet(
                name = cms.string('sigmaPhiPhiTot'),
                value = cms.string('sigmaPhiPhiTot()')
            ),
            cms.PSet(
                name = cms.string('sigmaRRTot'),
                value = cms.string('sigmaRRTot()')
            ),
            cms.PSet(
                name = cms.string('triggerCells90percent'),
                value = cms.string('triggerCells90percent()')
            ), 
            template = cms.PSetTemplate(
                name = cms.required.string,
                value = cms.required.string
            )
        ),
        weightsFile = cms.string('L1Trigger/Phase2L1ParticleFlow/data/hgcal_egID/Photon_Pion_vs_Neutrino_BDTweights_1116.xml.gz'),
        wp = cms.string('-0.10')
    ),
    hgcalTowers = cms.InputTag("l1tHGCalTowerProducer","HGCalTowerProcessor"),
    l1CaloTowers = cms.InputTag("l1tEGammaClusterEmuProducer","L1CaloTowerCollection"),
    mightGet = cms.optional.untracked.vstring,
    preEmId = cms.string('hOverE < 0.3 && hOverE >= 0')
)

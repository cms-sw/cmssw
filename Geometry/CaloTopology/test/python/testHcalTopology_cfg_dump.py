import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.source = cms.Source("EmptySource")
process.generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        PartID = cms.vint32(14),
        MaxEta = cms.double(3.5),
        MaxPhi = cms.double(3.14159265359),
        MinEta = cms.double(-3.5),
        MinE = cms.double(9.99),
        MinPhi = cms.double(-3.14159265359),
        MaxE = cms.double(10.01)
    ),
    Verbosity = cms.untracked.int32(0),
    AddAntiParticle = cms.bool(False),
    firstRun = cms.untracked.uint32(1)
)


process.randomEngineStateProducer = cms.EDProducer("RandomEngineStateProducer")


process.prod = cms.EDAnalyzer("HcalTopologyTester")


process.p1 = cms.Path(process.generator+process.prod)


process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('*'),
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        threshold = cms.untracked.string('DEBUG'),
        HCalGeom = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    ),
    categories = cms.untracked.vstring('HCalGeom'),
    destinations = cms.untracked.vstring('cout')
)


process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    horeco = cms.PSet(
        initialSeed = cms.untracked.uint32(541321),
        engineName = cms.untracked.string('TRandom3')
    ),
    externalLHEProducer = cms.PSet(
        initialSeed = cms.untracked.uint32(234567),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    saveFileName = cms.untracked.string(''),
    hbhereco = cms.PSet(
        initialSeed = cms.untracked.uint32(541321),
        engineName = cms.untracked.string('TRandom3')
    ),
    simSiStripDigiSimLink = cms.PSet(
        initialSeed = cms.untracked.uint32(1234567),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    paramMuons = cms.PSet(
        initialSeed = cms.untracked.uint32(54525),
        engineName = cms.untracked.string('TRandom3')
    ),
    famosPileUp = cms.PSet(
        initialSeed = cms.untracked.uint32(918273),
        engineName = cms.untracked.string('TRandom3')
    ),
    simMuonDTDigis = cms.PSet(
        initialSeed = cms.untracked.uint32(1234567),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    siTrackerGaussianSmearingRecHits = cms.PSet(
        initialSeed = cms.untracked.uint32(24680),
        engineName = cms.untracked.string('TRandom3')
    ),
    ecalPreshowerRecHit = cms.PSet(
        initialSeed = cms.untracked.uint32(6541321),
        engineName = cms.untracked.string('TRandom3')
    ),
    generator = cms.PSet(
        initialSeed = cms.untracked.uint32(456789),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    simMuonRPCDigis = cms.PSet(
        initialSeed = cms.untracked.uint32(1234567),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    hiSignal = cms.PSet(
        initialSeed = cms.untracked.uint32(123456789),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    mixSimCaloHits = cms.PSet(
        initialSeed = cms.untracked.uint32(918273),
        engineName = cms.untracked.string('TRandom3')
    ),
    mix = cms.PSet(
        initialSeed = cms.untracked.uint32(12345),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    VtxSmeared = cms.PSet(
        initialSeed = cms.untracked.uint32(98765432),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    LHCTransport = cms.PSet(
        initialSeed = cms.untracked.uint32(87654321),
        engineName = cms.untracked.string('TRandom3')
    ),
    ecalRecHit = cms.PSet(
        initialSeed = cms.untracked.uint32(654321),
        engineName = cms.untracked.string('TRandom3')
    ),
    mixData = cms.PSet(
        initialSeed = cms.untracked.uint32(12345),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    mixRecoTracks = cms.PSet(
        initialSeed = cms.untracked.uint32(918273),
        engineName = cms.untracked.string('TRandom3')
    ),
    hiSignalG4SimHits = cms.PSet(
        initialSeed = cms.untracked.uint32(11),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    famosSimHits = cms.PSet(
        initialSeed = cms.untracked.uint32(13579),
        engineName = cms.untracked.string('TRandom3')
    ),
    MuonSimHits = cms.PSet(
        initialSeed = cms.untracked.uint32(987346),
        engineName = cms.untracked.string('TRandom3')
    ),
    g4SimHits = cms.PSet(
        initialSeed = cms.untracked.uint32(11),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    hiSignalLHCTransport = cms.PSet(
        initialSeed = cms.untracked.uint32(88776655),
        engineName = cms.untracked.string('TRandom3')
    ),
    mixGenPU = cms.PSet(
        initialSeed = cms.untracked.uint32(918273),
        engineName = cms.untracked.string('TRandom3')
    ),
    l1ParamMuons = cms.PSet(
        initialSeed = cms.untracked.uint32(6453209),
        engineName = cms.untracked.string('TRandom3')
    ),
    simBeamSpotFilter = cms.PSet(
        initialSeed = cms.untracked.uint32(87654321),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    simMuonCSCDigis = cms.PSet(
        initialSeed = cms.untracked.uint32(11223344),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    hfreco = cms.PSet(
        initialSeed = cms.untracked.uint32(541321),
        engineName = cms.untracked.string('TRandom3')
    )
)


process.HcalRecNumberingInitialization = cms.ESProducer("HcalRecNumberingInitialization")


process.HcalSimNumberingInitialization = cms.ESProducer("HcalSimNumberingInitialization")


process.hcalTopologyIdeal = cms.ESProducer("HcalTopologyIdealEP",
    Exclude = cms.untracked.string(''),
    appendToDataLabel = cms.string(''),
    hcalTopologyConstants = cms.PSet(

    )
)


process.HepPDTESSource = cms.ESSource("HepPDTESSource",
    pdtFileName = cms.FileInPath('SimGeneral/HepPDTESSource/data/particle.tbl')
)


process.XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml', 
        'Geometry/HcalCommonData/data/average/hcalforwardmaterial.xml', 
        'Geometry/CMSCommonData/data/rotations.xml', 
        'Geometry/HcalCommonData/data/hcalrotations.xml', 
        'Geometry/CMSCommonData/data/normal/cmsextent.xml', 
        'Geometry/CMSCommonData/data/cms.xml', 
        'Geometry/CMSCommonData/data/cmsMother.xml', 
        'Geometry/CMSCommonData/data/caloBase.xml', 
        'Geometry/CMSCommonData/data/cmsCalo.xml', 
        'Geometry/CMSCommonData/data/muonBase.xml', 
        'Geometry/CMSCommonData/data/cmsMuon.xml', 
        'Geometry/CMSCommonData/data/mgnt.xml', 
        'Geometry/CMSCommonData/data/muonMB.xml', 
        'Geometry/CMSCommonData/data/muonMagnet.xml', 
        'Geometry/CMSCommonData/data/eta4/etaMax.xml', 
        'Geometry/HcalCommonData/data/Phase0/hcalalgo.xml', 
        'Geometry/HcalCommonData/data/hcalcablealgo.xml', 
        'Geometry/HcalCommonData/data/hcalbarrelalgo.xml', 
        'Geometry/HcalCommonData/data/Phase0/hcalendcapalgo.xml', 
        'Geometry/HcalCommonData/data/hcalouteralgo.xml', 
        'Geometry/HcalCommonData/data/hcalforwardalgo.xml', 
        'Geometry/HcalCommonData/data/Phase0/hcalSimNumbering.xml', 
        'Geometry/HcalCommonData/data/Phase0/hcalRecNumbering.xml', 
        'Geometry/ForwardCommonData/data/bundle/forwardshield.xml', 
        'Geometry/ForwardCommonData/data/forward.xml', 
        'Geometry/HcalSimData/data/CaloUtil.xml', 
        'Geometry/HcalCommonData/data/hcalsens.xml', 
        'Geometry/HcalSimData/data/hf.xml', 
        'Geometry/HcalSimData/data/hfpmt.xml', 
        'Geometry/HcalSimData/data/hffibrebundle.xml', 
        'Geometry/HcalSimData/data/HcalProdCuts.xml', 
        'Geometry/CMSCommonData/data/FieldParameters.xml'),
    rootNodeName = cms.string('cms:OCMS')
)


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)


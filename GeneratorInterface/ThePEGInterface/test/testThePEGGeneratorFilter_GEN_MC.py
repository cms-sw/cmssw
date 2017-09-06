import FWCore.ParameterSet.Config as cms

process = cms.Process('GEN')

process.load("Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

process.generator = cms.EDFilter("ThePEGGeneratorFilter",
    cm10TeV = cms.vstring('set /Herwig/Generators/LHCGenerator:EventHandler:LuminosityFunction:Energy 10000.0', 
        'set /Herwig/Shower/Evolver:IntrinsicPtGaussian 2.1*GeV'),
    run = cms.string('LHC'),
    repository = cms.string('HerwigDefaults.rpo'),
    cm14TeV = cms.vstring('set /Herwig/Generators/LHCGenerator:EventHandler:LuminosityFunction:Energy 14000.0', 
        'set /Herwig/Shower/Evolver:IntrinsicPtGaussian 2.2*GeV'),
    dataLocation = cms.string('${HERWIGPATH}'),
    lheDefaults = cms.vstring('cd /Herwig/Cuts', 
        'create ThePEG::Cuts NoCuts', 
        'cd /Herwig/EventHandlers', 
        'create ThePEG::LesHouchesInterface LHEReader', 
        'set LHEReader:Cuts /Herwig/Cuts/NoCuts', 
        'create ThePEG::LesHouchesEventHandler LHEHandler', 
        'set LHEHandler:WeightOption VarWeight', 
        'set LHEHandler:PartonExtractor /Herwig/Partons/QCDExtractor', 
        'set LHEHandler:CascadeHandler /Herwig/Shower/ShowerHandler', 
        'set LHEHandler:HadronizationHandler /Herwig/Hadronization/ClusterHadHandler', 
        'set LHEHandler:DecayHandler /Herwig/Decays/DecayHandler', 
        'insert LHEHandler:LesHouchesReaders 0 LHEReader', 
        'cd /Herwig/Generators', 
        'set LHCGenerator:EventHandler /Herwig/EventHandlers/LHEHandler', 
        'cd /Herwig/Shower', 
        'set Evolver:HardVetoScaleSource Read', 
        'set Evolver:MECorrMode No', 
        'cd /'),
    cmsDefaults = cms.vstring(
        '+pdfCTEQ6LL',
        '+basicSetup', 
        '+cm14TeV', 
        '+setParticlesStableForDetector'),
    lheDefaultPDFs = cms.vstring('cd /Herwig/EventHandlers', 
        'set LHEReader:PDFA /cmsPDFSet', 
        'set LHEReader:PDFB /cmsPDFSet', 
        'cd /'),
    reweightPthat = cms.vstring('mkdir /Herwig/Weights', 
        'cd /Herwig/Weights', 
        'create ThePEG::ReweightMinPT reweightMinPT ReweightMinPT.so', 
        'cd /', 
        'set /Herwig/Weights/reweightMinPT:Power 4.5', 
        'set /Herwig/Weights/reweightMinPT:Scale 15*GeV', 
        'insert SimpleQCD:Reweights[0] /Herwig/Weights/reweightMinPT'),
    generatorModule = cms.string('/Herwig/Generators/LHCGenerator'),
    eventHandlers = cms.string('/Herwig/EventHandlers'),
    basicSetup = cms.vstring('cd /Herwig/Generators', 
        'create ThePEG::RandomEngineGlue /Herwig/RandomGlue', 
        'set LHCGenerator:RandomNumberGenerator /Herwig/RandomGlue', 
        'set LHCGenerator:NumberOfEvents 10000000', 
        'set LHCGenerator:DebugLevel 1', 
        'set LHCGenerator:PrintEvent 0', 
        'set LHCGenerator:MaxErrors 10000'),
    setParticlesStableForDetector = cms.vstring('cd /Herwig/Particles', 
        'set mu-:Stable Stable', 
        'set mu+:Stable Stable', 
        'set Sigma-:Stable Stable', 
        'set Sigmabar+:Stable Stable', 
        'set Lambda0:Stable Stable', 
        'set Lambdabar0:Stable Stable', 
        'set Sigma+:Stable Stable', 
        'set Sigmabar-:Stable Stable', 
        'set Xi-:Stable Stable', 
        'set Xibar+:Stable Stable', 
        'set Xi0:Stable Stable', 
        'set Xibar0:Stable Stable', 
        'set Omega-:Stable Stable', 
        'set Omegabar+:Stable Stable', 
        'set pi+:Stable Stable', 
        'set pi-:Stable Stable', 
        'set K+:Stable Stable', 
        'set K-:Stable Stable', 
        'set K_S0:Stable Stable', 
        'set K_L0:Stable Stable', 
        'cd /'),
    pdfCTEQ6LL = cms.vstring(
        'cd /Herwig/Partons',
        'create ThePEG::LHAPDF myPDFset ThePEGLHAPDF.so',
        'set myPDFset:PDFName cteq6ll',
        'set myPDFset:RemnantHandler HadronRemnants',
        'set /Herwig/Particles/p+:PDF myPDFset',
        'set /Herwig/Particles/pbar-:PDF myPDFset',
        'cd /'),
    pdfCT10 = cms.vstring(
        'cd /Herwig/Partons',
        'create ThePEG::LHAPDF myPDFset ThePEGLHAPDF.so',
        'set myPDFset:PDFName CT10',
        'set myPDFset:RemnantHandler HadronRemnants',
        'set /Herwig/Particles/p+:PDF myPDFset',
        'set /Herwig/Particles/pbar-:PDF myPDFset',
        'cd /'),
    cm7TeV = cms.vstring('set /Herwig/Generators/LHCGenerator:EventHandler:LuminosityFunction:Energy 7000.0', 
        'set /Herwig/Shower/Evolver:IntrinsicPtGaussian 2.0*GeV'),
    reweightConstant = cms.vstring('mkdir /Herwig/Weights', 
        'cd /Herwig/Weights', 
        'create ThePEG::ReweightConstant reweightConstant ReweightConstant.so', 
        'cd /', 
        'set /Herwig/Weights/reweightConstant:C 1', 
        'insert SimpleQCD:Reweights[0] /Herwig/Weights/reweightConstant'),
    eventsToPrint = cms.untracked.uint32(1),
    dumpConfig = cms.untracked.string('dump.config'),
    dumpEvents = cms.untracked.string('dump.hepmc'),
    validationQCD = cms.vstring('cd /Herwig/MatrixElements/', 
        'insert SimpleQCD:MatrixElements[0] MEQCD2to2', 
        'cd /', 
        'set /Herwig/Cuts/JetKtCut:MinKT 50*GeV', 
        'set /Herwig/Cuts/JetKtCut:MaxKT 100*GeV'),
    validationMSSM = cms.vstring('cd /Herwig/NewPhysics', 
        'set HPConstructor:IncludeEW No', 
        'set TwoBodyDC:CreateDecayModes No', 
        'setup MSSM/Model ${HERWIGPATH}/SPhenoSPS1a.spc', 
        'insert NewModel:DecayParticles 0 /Herwig/Particles/~d_L', 
        'insert NewModel:DecayParticles 1 /Herwig/Particles/~u_L', 
        'insert NewModel:DecayParticles 2 /Herwig/Particles/~e_R-', 
        'insert NewModel:DecayParticles 3 /Herwig/Particles/~mu_R-', 
        'insert NewModel:DecayParticles 4 /Herwig/Particles/~chi_10', 
        'insert NewModel:DecayParticles 5 /Herwig/Particles/~chi_20', 
        'insert NewModel:DecayParticles 6 /Herwig/Particles/~chi_2+'),
    configFiles = cms.vstring(),
    parameterSets = cms.vstring('cmsDefaults', 
        'validationQCD')
)


process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(2)
        )
    ),
    destinations = cms.untracked.vstring('cout')
)

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    generator = cms.PSet(
        initialSeed = cms.untracked.uint32(123456789),
    )
)


process.GEN = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testThePEGGeneratorFilter_GEN.root')
)

process.p = cms.Path(process.generator)
process.outpath = cms.EndPath(process.GEN)

process.schedule = cms.Schedule(process.p, process.outpath)

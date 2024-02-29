import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.load("Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff")

process.source = cms.Source("EmptySource")

process.generator = cms.EDFilter("Pythia8GeneratorFilter",
    maxEventsToPrint = cms.untracked.int32(1),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    filterEfficiency = cms.untracked.double(1.0),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    comEnergy = cms.double(5020.),
    doProtonPhotonFlux = cms.untracked.bool(True),
    #PPbarInitialState = cms.PSet(),
    #SLHAFileForPythia8 = cms.string('Configuration/Generator/data/CSA07SUSYBSM_LM9p_sftsdkpyt_slha.out'),
    #reweightGen = cms.PSet( # flat in pT
    #   pTRef = cms.double(15.0),
    #   power = cms.double(4.5)
    #),
    #reweightGenRap = cms.PSet( # flat in eta
    #   yLabSigmaFunc = cms.string("15.44/pow(x,0.0253)-12.56"),
    #   yLabPower = cms.double(2.),
    #   yCMSigmaFunc = cms.string("5.45/pow(x+64.84,0.34)"),
    #   yCMPower = cms.double(2.),
    #   pTHatMin = cms.double(15.),
    #   pTHatMax = cms.double(3000.)
    #),
    #reweightGenPtHatRap = cms.PSet( # flat in Pt and eta
    #   yLabSigmaFunc = cms.string("15.44/pow(x,0.0253)-12.56"),
    #   yLabPower = cms.double(2.),
    #   yCMSigmaFunc = cms.string("5.45/pow(x+64.84,0.34)"),
    #   yCMPower = cms.double(2.),
    #   pTHatMin = cms.double(15.),
    #   pTHatMax = cms.double(3000.)
    #),
    PythiaParameters = cms.PSet(
        pythia8_example02 = cms.vstring('HardQCD:all = on',
                                        'PhaseSpace:pTHatMin = 10.',#CM Edit 20->10
                                        'PhotonParton:all = on',#Added from main70
                                        'MultipartonInteractions:pT0Ref = 3.0',#Added from main70
                                        'PDF:lepton2gamma = on',#Added from main70
                                        #This option below crashes - debug
                                        'PDF:lepton2gammaSet = 2',#Added from main70
                                        'PDF:useHardNPDFB = on',
                                        'Photon:sampleQ2 = off'
                                    ), 
        parameterSets = cms.vstring('pythia8_example02')
    )
)

# in order to use lhapdf PDF add a line like this to pythia8_example02:
# 'PDF:pSet = LHAPDF6:CT10'

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    generator = cms.PSet(
        initialSeed = cms.untracked.uint32(123456789),
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.GEN = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('pythia8ex2.root')
)

process.p = cms.Path(process.generator)
process.outpath = cms.EndPath(process.GEN)

process.schedule = cms.Schedule(process.p, process.outpath)

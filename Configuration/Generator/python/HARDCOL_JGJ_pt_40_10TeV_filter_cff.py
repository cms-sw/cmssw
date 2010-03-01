import FWCore.ParameterSet.Config as cms

source = cms.Source("MCDBSource",
        articleID = cms.uint32(290),
        supportedProtocols = cms.untracked.vstring('rfio')
        #filter = cms.untracked.string('\\.lhe$')
)

from Configuration.Generator.PythiaUESettings_cfi import *

generator = cms.EDProducer("Pythia6HadronizerFilter",
    pythiaHepMCVerbosity = cms.untracked.bool(True),
    maxEventsToPrint = cms.untracked.int32(0),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    comEnergy = cms.double(10000.0),
    PythiaParameters = cms.PSet(
        pythiaUESettingsBlock,
        processParameters = cms.vstring('MSEL=0         ! User defined processes', 
                        'PMAS(5,1)=4.4   ! b quark mass',
                        'PMAS(6,1)=172.4 ! t quark mass',
			'MSTJ(1)=1       ! Fragmentation/hadronization on or off',
			'MSTP(61)=1      ! Parton showering on or off'),
        # This is a vector of ParameterSet names to be read, in this order
        parameterSets = cms.vstring('pythiaUESettings', 
            'processParameters')
    )
)

genParticlesjgj = cms.EDProducer("GenParticleProducer",
    saveBarCodes = cms.untracked.bool(True),
    src = cms.InputTag("generator"),
    abortOnUnknownPDGCode = cms.untracked.bool(True)
)

genJetParticlesjgj = cms.EDProducer("InputGenJetsParticleSelector",
    src = cms.InputTag("genParticlesjgj"),
    ignoreParticleIDs = cms.vuint32(1000022, 2000012, 2000014, 2000016, 1000039, 
        5000039, 4000012, 9900012, 9900014, 9900016, 
        39),
    partonicFinalState = cms.bool(False),
    excludeResonances = cms.bool(True),
    excludeFromResonancePids = cms.vuint32(12, 13, 14, 16),
    tausAsJets = cms.bool(False)
)

GenJetParametersjgj = cms.PSet(
    src            = cms.InputTag("genJetParticlesjgj"),
    srcPVs         = cms.InputTag(''),
    jetType        = cms.string('GenJet'),
    jetPtMin       = cms.double(5.0),
    inputEtMin     = cms.double(0.0),
    inputEMin      = cms.double(0.0),
    doPVCorrection = cms.bool(False),
    # pileup with offset correction
    doPUOffsetCorr = cms.bool(False),
       # if pileup is false, these are not read:
       nSigmaPU = cms.double(1.0),
       radiusPU = cms.double(0.5),  
    # fastjet-style pileup     
    doPUFastjet    = cms.bool(False),
      # if doPU is false, these are not read:
      Active_Area_Repeats = cms.int32(5),
      GhostArea = cms.double(0.01),
      Ghost_EtaMax = cms.double(6.0)
)

AnomalousCellParametersjgj = cms.PSet(
    maxBadEcalCells         = cms.uint32(9999999),
    maxRecoveredEcalCells   = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxBadHcalCells         = cms.uint32(9999999),
    maxRecoveredHcalCells   = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999)
)

kt4GenJetsjgj = cms.EDProducer(
    "FastjetJetProducer",
    GenJetParametersjgj,
    AnomalousCellParametersjgj,
    jetAlgorithm = cms.string("Kt"),
    rParam       = cms.double(0.4)
)

kt6GenJetsjgj = kt4GenJetsjgj.clone( rParam = 0.6 )

sisCone5GenJetsjgj = cms.EDProducer(
    "FastjetJetProducer",
    GenJetParametersjgj,
    AnomalousCellParametersjgj,
    jetAlgorithm = cms.string("SISCone"),
    rParam       = cms.double(0.5)
)


#genJetParticlesjgj = cms.Sequence(genParticlesForJetsjgj)
#recoGenJetsjgj = cms.Sequence(sisCone5GenJetsjgj + kt4GenJetsjgj)

#genJetjgj = cms.Sequence(genParticlesjgj + (genJetParticlesjgj + recoGenJetsjgj))

jgjFilter = cms.EDFilter('JGJFilter')

ProductionFilterSequence = cms.Sequence(generator*(genParticlesjgj+(genJetParticlesjgj+(sisCone5GenJetsjgj+kt4GenJetsjgj+kt6GenJetsjgj)))*jgjFilter)

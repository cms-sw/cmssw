import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
#process.load("SimGeneral.HepPDTESSource.pdt_cfi")
process.load("GeneratorInterface.ExternalDecays.TauSpinner_cfi")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
                                                   generator = cms.PSet(initialSeed = cms.untracked.uint32(123456789),
                                                                        engineName = cms.untracked.string('HepJamesRandom')
                                                                        ),
                                                   TauSpinnerGen  = cms.PSet(initialSeed = cms.untracked.uint32(123456789),
                                                                             engineName = cms.untracked.string('HepJamesRandom')
                                                                             )
                                                   )

process.randomEngineStateProducer = cms.EDProducer("RandomEngineStateProducer")

process.Timing=cms.Service("Timing",
                           summaryOnly=cms.untracked.bool(True))


# The following three lines reduce the clutter of repeated printouts
# of the same exception message.
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.destinations = ['cerr']
process.MessageLogger.statistics = []
process.MessageLogger.fwkJobReports = []

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(100))

process.source = cms.Source("EmptySource")

from Configuration.Generator.PythiaUESettings_cfi import *
process.generator = cms.EDFilter("Pythia6GeneratorFilter",
                                 pythiaHepMCVerbosity = cms.untracked.bool(True),
                                 maxEventsToPrint = cms.untracked.int32(0),
                                 pythiaPylistVerbosity = cms.untracked.int32(1),
                                     # this shows how to turn ON some of the general Py6 printouts, like banner...
                                 ## --> displayPythiaBanner = cms.untracked.bool(True),
                                 ## --> displayPythiaCards = cms.untracked.bool(True),
                                 comEnergy = cms.double(7000.0),
                                 
                                 ExternalDecays = cms.PSet(
    Tauola = cms.untracked.PSet(
    # these settings below exemplfy how to use "native" Tauola approach:
    # one MUST set mdtau=1 !!! then pjak1 & pjak2 will translate into
    # native variables jak1 & jak2 (jak1/jak2=4 means that both tau's
    # decay into the rho-mode
    #
    UseTauolaPolarization = cms.bool(True),
    InputCards = cms.PSet
    (
    pjak1 = cms.int32(0),
    pjak2 = cms.int32(0),
    mdtau = cms.int32(0) # generic tau decays
    ## mdtau = cms.int32(240) # (any) tau -> nu pi+-
    )
    #           TauolaDefaultInputCards,
    #          TauolaPolar
    ),
    parameterSets = cms.vstring('Tauola')
    ),
                                 PythiaParameters = cms.PSet(
    pythiaUESettingsBlock,
    processParameters = cms.vstring('MSEL=0            !User defined processes',
                                    'MSUB(1)=1         !Incl Z0/gamma* production',
                                    'MSTP(43)=3        !Both Z0 and gamma*',
                                    'MDME(174,1)=0     !Z decay into d dbar',
                                    'MDME(175,1)=0     !Z decay into u ubar',
                                    'MDME(176,1)=0     !Z decay into s sbar',
                                    'MDME(177,1)=0     !Z decay into c cbar',
                                    'MDME(178,1)=0     !Z decay into b bbar',
                                    'MDME(179,1)=0     !Z decay into t tbar',
                                    'MDME(182,1)=0     !Z decay into e- e+',
                                    'MDME(183,1)=0     !Z decay into nu_e nu_ebar',
                                    'MDME(184,1)=0     !Z decay into mu- mu+',
                                    'MDME(185,1)=0     !Z decay into nu_mu nu_mubar',
                                    'MDME(186,1)=1     !Z decay into tau- tau+',
                                    'MDME(187,1)=0     !Z decay into nu_tau nu_taubar',
                                    'CKIN(1)=50.       !Minimum sqrt(s_hat) value (=Z mass)'),
    # This is a vector of ParameterSet names to be read, in this order
    parameterSets = cms.vstring('pythiaUESettings',
                                'processParameters')
    )
                                 )

# Produce PDF weights (maximum is 3)
process.pdfWeights = cms.EDProducer("PdfWeightProducer",
                                    # Fix POWHEG if buggy (this PDF set will also appear on output,
                                    # so only two more PDF sets can be added in PdfSetNames if not "")
                                    #FixPOWHEG = cms.untracked.string("cteq66.LHgrid"),
                                    #GenTag = cms.untracked.InputTag("genParticles"),
                                    PdfInfoTag = cms.untracked.InputTag("VtxSmeared"),
                                    PdfSetNames = cms.untracked.vstring(
    #    "cteq66.LHgrid"
    #    , "MRST2006nnlo.LHgrid" ,
        "MSTW2008nnlo90cl.LHgrid"
        )
                                    )


process.p1 = cms.Path( process.TauSpinnerGen )

process.GEN = cms.OutputModule("PoolOutputModule",
                               fileName = cms.untracked.string('Test_Py6_Z2TauTau_Tauola.root')
                               )

process.p = cms.Path(process.generator)

process.outpath = cms.EndPath(process.GEN)
process.p1 = cms.Path(process.randomEngineStateProducer*process.TauSpinnerGen)
process.schedule = cms.Schedule(process.p, process.p1, process.outpath)


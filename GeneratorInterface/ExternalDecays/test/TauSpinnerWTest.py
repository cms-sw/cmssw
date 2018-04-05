import FWCore.ParameterSet.Config as cms

#
# WARNING: This is NOT an example for users -
#          it's my private (JY) "development" cfg, for testing
#          newly implemented PhotosInterface - which is NOT yet
#          released via ExternalDecayDeriver
#

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
    processParameters = cms.vstring('MSEL        = 0    !User defined processes',
                                    'MSUB(2)     = 1    !W production',
                                    'MDME(190,1) = 0    !W decay into dbar u',
                                    'MDME(191,1) = 0    !W decay into dbar c',
                                    'MDME(192,1) = 0    !W decay into dbar t',
                                    'MDME(194,1) = 0    !W decay into sbar u',
                                    'MDME(195,1) = 0    !W decay into sbar c',
                                    'MDME(196,1) = 0    !W decay into sbar t',
                                    'MDME(198,1) = 0    !W decay into bbar u',
                                    'MDME(199,1) = 0    !W decay into bbar c',
                                    'MDME(200,1) = 0    !W decay into bbar t',
                                    'MDME(205,1) = 0    !W decay into bbar tp',
                                    'MDME(206,1) = 0    !W decay into e+ nu_e',
                                    'MDME(207,1) = 0    !W decay into mu+ nu_mu',
                                    'MDME(208,1) = 1    !W decay into tau+ nu_tau'),
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
                               fileName = cms.untracked.string('Test_Py6_W2TauNu_Tauola.root')
                               )

process.p = cms.Path(process.generator)

process.outpath = cms.EndPath(process.GEN)
process.p1 = cms.Path(process.randomEngineStateProducer*process.TauSpinnerGen)
process.schedule = cms.Schedule(process.p, process.p1, process.outpath)


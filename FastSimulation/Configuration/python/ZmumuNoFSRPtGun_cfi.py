

# The following comments couldn't be translated into the new config version:

# Single tau(+decays) ptgun

import FWCore.ParameterSet.Config as cms

#process = cms.Process("Gen")
# this example configuration offers some minimum
# annotation, to help users get through; please
# don't hesitate to read through the comments
# use MessageLogger to redirect/suppress multiple
# service messages coming from the system
#
# in this config below, we use the replace option to make
# the logger let out messages of severity ERROR (INFO level
# will be suppressed), and we want to limit the number to 10
#
#process.load("FWCore.MessageService.MessageLogger_cfi")

#process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

# Event output
#process.load("Configuration.EventContent.EventContent_cff")

#process.maxEvents = cms.untracked.PSet(
#        input = cms.untracked.int32(5)
#        )

RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
                                           generator = cms.PSet(
        initialSeed = cms.untracked.uint32(123456789),
        engineName = cms.untracked.string('HepJamesRandom')
        ),
                                           famosPileUp = cms.PSet(
        initialSeed = cms.untracked.uint32(543212345),
        engineName = cms.untracked.string('HepJamesRandom')
        ),
                                           famosSimHits = cms.PSet(
        initialSeed = cms.untracked.uint32(123412552),
        engineName = cms.untracked.string('HepJamesRandom')
        ),
                                           MuonSimHits = cms.PSet(
        initialSeed = cms.untracked.uint32(132412351),
        engineName = cms.untracked.string('HepJamesRandom')
        ),
                                           mix = cms.PSet(
        initialSeed = cms.untracked.uint32(132456),
        engineName = cms.untracked.string('HepJamesRandom')
        ),
                                           ecalPreshowerRecHit = cms.PSet(
        initialSeed = cms.untracked.uint32(123516413),
        engineName = cms.untracked.string('HepJamesRandom')
        ),
                                           hbhereco = cms.PSet(
        initialSeed = cms.untracked.uint32(1424512),
        engineName = cms.untracked.string('HepJamesRandom')
        ),
                                           horeco = cms.PSet(
        initialSeed = cms.untracked.uint32(12351346),
        engineName = cms.untracked.string('HepJamesRandom')
        ),
                                           hfreco = cms.PSet(
        initialSeed = cms.untracked.uint32(123463),
        engineName = cms.untracked.string('HepJamesRandom')
        ),
                                           simMuonCSCDigis = cms.PSet(
        initialSeed = cms.untracked.uint32(123462),
        engineName = cms.untracked.string('HepJamesRandom')
        ),
                                           simMuonDTDigis = cms.PSet(
        initialSeed = cms.untracked.uint32(1234123),
        engineName = cms.untracked.string('HepJamesRandom')
        ),
                                           simMuonRPCDigis = cms.PSet(
        initialSeed = cms.untracked.uint32(1234),
        engineName = cms.untracked.string('HepJamesRandom')
        ),
                                           siTrackerGaussianSmearingRecHits = cms.PSet(
        initialSeed = cms.untracked.uint32(124),
        engineName = cms.untracked.string('HepJamesRandom')
        )





                                           )


source = cms.Source("EmptySource")

generator = cms.EDProducer("Pythia6PtGun",
                                       maxEventsToPrint = cms.untracked.int32(5),
                                       pythiaPylistVerbosity = cms.untracked.int32(1),
                                       pythiaHepMCVerbosity = cms.untracked.bool(True),
                                       PGunParameters = cms.PSet(
            ParticleID = cms.vint32(23),
                    AddAntiParticle = cms.bool(False),
                    MinPhi = cms.double(-3.14159265359),
                    MaxPhi = cms.double(3.14159265359),
                    MinPt = cms.double( 0.0),
                    MaxPt = cms.double(50.0),
                    MinEta = cms.double(0),
                    MaxEta = cms.double(4.2)
                ),
                                       PythiaParameters = cms.PSet(
            pythiaDefault = cms.vstring(
       'PMAS(5,1)=4.8 ! b quark mass',
                  'PMAS(6,1)=172.3 ! t quark mass',
                  'MSTP(61)=0 ! initial state radiation',
          'mstj(41)=1' # per Steve M., instead of mstp(71)... btw, shoult it be 0 or 1 ?
                  #'MSTP(71)=0 !final state radiation'
       ),
            pythiaZtoMuons = cms.vstring(
       "MDME(174,1)=0",          # !Z decay into d dbar,
                  "MDME(175,1)=0",          # !Z decay into u ubar,
                  "MDME(176,1)=0",          # !Z decay into s sbar,
                  "MDME(177,1)=0",          # !Z decay into c cbar,
                  "MDME(178,1)=0",          # !Z decay into b bbar,
                  "MDME(179,1)=0",          # !Z decay into t tbar,
                  "MDME(182,1)=0",          # !Z decay into e- e+,
                  "MDME(183,1)=0",          # !Z decay into nu_e nu_ebar,
                  "MDME(184,1)=1",          # !Z decay into mu- mu+,
                  "MDME(185,1)=0",          # !Z decay into nu_mu nu_mubar,
                  "MDME(186,1)=0",          # !Z decay into tau- tau+,
                  "MDME(187,1)=0"           # !Z decay into nu_tau nu_taubar
               ),
                    parameterSets = cms.vstring(
                'pythiaDefault',
                            'pythiaZtoMuons'
                        )
                )
                                   )

#process.FEVT = cms.OutputModule("PoolOutputModule",
#                                    process.FEVTSIMEventContent,
#                                    fileName = cms.untracked.string('gen_singleTau.root')
#                                )

#process.p = cms.Path(process.generator)
#process.outpath = cms.EndPath(process.FEVT)
#process.schedule = cms.Schedule(process.p,process.outpath)



ProductionFilterSequence = cms.Sequence(generator)

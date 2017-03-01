import FWCore.ParameterSet.Config as cms
from Configuration.Generator.Pythia8CommonSettings_cfi import *
from Configuration.Generator.Pythia8CUEP8M1Settings_cfi import *

generator = cms.EDFilter("Pythia8GeneratorFilter",
                         pythiaPylistVerbosity = cms.untracked.int32(0),
                         filterEfficiency = cms.untracked.double(0.53),
                         pythiaHepMCVerbosity = cms.untracked.bool(False),
                         crossSection = cms.untracked.double(9090000.0),
                         comEnergy = cms.double(13000.0),
                         maxEventsToPrint = cms.untracked.int32(0),
                         PythiaParameters = cms.PSet(
        pythia8CommonSettingsBlock,
        pythia8CUEP8M1SettingsBlock,
        processParameters = cms.vstring(
            'Bottomonium:states(3S1) = 553', # filter on 553 and prevents other onium states decaying to 553, so we should turn the others off
            'Bottomonium:O(3S1)[3S1(1)] = 9.28',
            'Bottomonium:O(3S1)[3S1(8)] = 0.15',
            'Bottomonium:O(3S1)[1S0(8)] = 0.02',
            'Bottomonium:O(3S1)[3P0(8)] = 0.02',
            'Bottomonium:gg2bbbar(3S1)[3S1(1)]g = on',
            'Bottomonium:gg2bbbar(3S1)[3S1(8)]g = on',
            'Bottomonium:qg2bbbar(3S1)[3S1(8)]q = on',
            'Bottomonium:qqbar2bbbar(3S1)[3S1(8)]g = on',
            'Bottomonium:gg2bbbar(3S1)[1S0(8)]g = on',
            'Bottomonium:qg2bbbar(3S1)[1S0(8)]q = on',
            'Bottomonium:qqbar2bbbar(3S1)[1S0(8)]g = on',
            'Bottomonium:gg2bbbar(3S1)[3PJ(8)]g = on',
            'Bottomonium:qg2bbbar(3S1)[3PJ(8)]q = on',
            'Bottomonium:qqbar2bbbar(3S1)[3PJ(8)]g = on',
            '553:onMode = off',            # ignore cross-section re-weighting (CSAMODE=6) since selecting wanted decay mode
            '553:onIfAny = 13',
            'PhaseSpace:pTHatMin = 20.',
            ),
        parameterSets = cms.vstring('pythia8CommonSettings',
                                    'pythia8CUEP8M1Settings',
                                    'processParameters',
                                    )
        )
                         )

oniafilter = cms.EDFilter("PythiaFilter",
                          Status = cms.untracked.int32(2),
                          MaxEta = cms.untracked.double(1000.0),
                          MinEta = cms.untracked.double(-1000.0),
                          MinPt = cms.untracked.double(0.0),
                          ParticleID = cms.untracked.int32(553)
                          )

mumugenfilter = cms.EDFilter("MCParticlePairFilter",
                             Status = cms.untracked.vint32(1, 1),
                             MinPt = cms.untracked.vdouble(0.5, 0.5),
                             MinP = cms.untracked.vdouble(2.7, 2.7),
                             MaxEta = cms.untracked.vdouble(2.5, 2.5),
                             MinEta = cms.untracked.vdouble(-2.5, -2.5),
                             MinInvMass = cms.untracked.double(5.0),
                             MaxInvMass = cms.untracked.double(20.0),
                             ParticleCharge = cms.untracked.int32(-1),
                             ParticleID1 = cms.untracked.vint32(13),
                             ParticleID2 = cms.untracked.vint32(13)
                             )

ProductionFilterSequence = cms.Sequence(generator*oniafilter*mumugenfilter)

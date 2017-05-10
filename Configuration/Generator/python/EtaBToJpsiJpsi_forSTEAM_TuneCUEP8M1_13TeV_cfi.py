# For BPH trigger study only
import FWCore.ParameterSet.Config as cms
from Configuration.Generator.Pythia8CommonSettings_cfi import *
from Configuration.Generator.Pythia8CUEP8M1Settings_cfi import *
generator = cms.EDFilter("Pythia8GeneratorFilter",
maxEventsToPrint = cms.untracked.int32(1),
pythiaPylistVerbosity = cms.untracked.int32(1),
filterEfficiency = cms.untracked.double(1.0),
pythiaHepMCVerbosity = cms.untracked.bool(False),
comEnergy = cms.double(13000.),
PythiaParameters = cms.PSet(
pythia8CommonSettingsBlock,
pythia8CUEP8M1SettingsBlock,
pythiaEtab = cms.vstring(
'Higgs:useBSM = on',
'HiggsBSM:gg2H2 = on',
'HiggsH2:coup2d = 10.0',
'HiggsH2:coup2u = 10.0',
'HiggsH2:coup2Z = 0.0',
'HiggsH2:coup2W = 0.0',
'HiggsA3:coup2H2Z = 0.0',
'HiggsH2:coup2A3A3 = 0.0',
'HiggsH2:coup2H1H1 = 0.0',
'443:onMode = off',
'443:onIfMatch 13 13',
'333:onMode = off',
'333:onIfMatch 13 13',
'553:onMode = off',
'553:onIfMatch 13 13',
############# For Fixed Mass Distribution#############
'35:mMin = 0',
'35:mMax = 50.0',
'35:m0 = 7.0',
'35:mWidth = 0.00',
'35:addChannel 1 1.00 100 443 443',
'35:onMode = off',
'35:onIfMatch 443 443'), ## JpsiJpsi
# This is a vector of ParameterSet names to be read, in this order
parameterSets = cms.vstring('pythia8CommonSettings',
'pythia8CUEP8M1Settings',
'pythiaEtab')
)
)
etafilter = cms.EDFilter("PythiaFilter",
MaxEta = cms.untracked.double(9999.0),
MinEta = cms.untracked.double(-9999.0),
ParticleID = cms.untracked.int32(35)
)
jpsifilter = cms.EDFilter("PythiaDauVFilter",
MotherID = cms.untracked.int32(10551),
verbose = cms.untracked.int32(0),
ParticleID = cms.untracked.int32(443),
MaxEta = cms.untracked.vdouble(2.6, 2.6),
MinEta = cms.untracked.vdouble(-2.6, -2.6),
DaughterIDs = cms.untracked.vint32(13, -13),
MinPt = cms.untracked.vdouble(1.8, 1.8),
NumberDaughters = cms.untracked.int32(2)
)
ProductionFilterSequence = cms.Sequence(generator*etafilter)

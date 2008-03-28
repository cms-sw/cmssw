import FWCore.ParameterSet.Config as cms

# Magnetic Field
#
from MagneticField.Engine.volumeBasedMagneticField_cfi import *
from SimGeneral.HepPDTESSource.pythiapdt_cfi import *
# Geant4-based CMS Det.Simulation
#
# ECAL test beam specific OscarProducer configuration
from SimG4Core.Application.g4SimHits_cfi import *
# Test Beam ECAL specific MC info
#
SimEcalTBG4Object = cms.EDProducer("EcalTBMCInfoProducer",
    common_beam_direction_parameters,
    CrystalMapFile = cms.FileInPath('Geometry/EcalTestBeam/data/BarrelSM1CrystalCenterElectron120GeV.dat'),
    moduleLabelVtx = cms.untracked.string('source')
)

# Test Beam ECAL hodoscope raw data simulation
# 
SimEcalTBHodoscope = cms.EDProducer("TBHodoActiveVolumeRawInfoProducer")

# Test Beam ECAL Event header filling
# 
SimEcalEventHeader = cms.EDProducer("FakeTBEventHeaderProducer",
    EcalTBInfoLabel = cms.untracked.string('SimEcalTBG4Object')
)

g4SimHits.UseMagneticField = False
g4SimHits.Physics.DefaultCutValue = 1.
g4SimHits.Generator = cms.PSet(
    HectorEtaCut,
    ApplyPtCuts = cms.bool(False),
    MaxPhiCut = cms.double(3.14159265359), ## according to CMS conventions

    ApplyEtaCuts = cms.bool(True),
    MaxPtCut = cms.double(99999.0),
    MinPtCut = cms.double(0.001),
    ApplyPhiCuts = cms.bool(False),
    Verbosity = cms.untracked.int32(0),
    MinPhiCut = cms.double(-3.14159265359), ## in radians

    MaxEtaCut = cms.double(1.5),
    HepMCProductLabel = cms.string('source'),
    MinEtaCut = cms.double(0.0),
    DecLenCut = cms.double(10.0) ## the minimum decay length in cm (!) for mother tracking

)
g4SimHits.CaloSD.CorrectTOFBeam = True
g4SimHits.CaloSD.BeamPosition = -26733.5
g4SimHits.CaloTrkProcessing.TestBeam = True


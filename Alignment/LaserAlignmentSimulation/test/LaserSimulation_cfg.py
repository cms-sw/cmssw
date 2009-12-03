# The following comments couldn't be translated into the new config version:

#Geometry
# add the description of the Alignment Tubes

import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")
# include default services, like RandomNumberGenerator
process.load("Configuration.StandardSequences.Services_cff")

process.load("Configuration.StandardSequences.Geometry_cff")

process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("Configuration.StandardSequences.FakeConditions_cff")

# the following cff file is needed in CMSSW_2_0_X ... hopefully we can remove it later on...
#include "SimCalorimetry/HcalTrigPrimProducers/data/hcaltpdigi.cff"
#
process.load("Configuration.StandardSequences.VtxSmearedGauss_cff")

process.load("Configuration.StandardSequences.MixingNoPileUp_cff")

process.load("Configuration.StandardSequences.Simulation_cff")

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cout'),
    cout = cms.untracked.PSet(
        DDLParser = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        EcalGeom = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        default = cms.untracked.PSet( ## kill all messages in the log

            limit = cms.untracked.int32(0)
        ),
        TrackerSimInfoNumbering = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        CaloSim = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        GeometryInfo = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        FwkJob = cms.untracked.PSet( ## except *all* of FwkJob's      

            limit = cms.untracked.int32(-1)
        ),
        HCalGeom = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        HcalSim = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        TrackerMapDDDtoID = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    ),
    categories = cms.untracked.vstring('CaloSim', 
        'DDLParser', 
        'EcalGeom', 
        'FwkJob', 
        'GeometryInfo', 
        'HCalGeom', 
        'HCalSim', 
        'TrackerMapDDDtoID', 
        'TrackerSimInfoNumbering'),
    fwkJobReports = cms.untracked.vstring('FrameworkJobReport.xml')
)

process.source = cms.Source("LaserAlignmentSource",
    firstRun = cms.untracked.uint32(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
process.o1 = cms.OutputModule("PoolOutputModule",
    #untracked string logicalFileName = "LaserEvents.SIM-DIGI.root"
    #untracked string catalog = "PoolFileCatalog.xml"
    compressionLevel = cms.untracked.int32(9),
    fileName = cms.untracked.string('LaserEvents.SIM-DIGI.root')
)

process.p1 = cms.Path(process.simulation)
process.output = cms.EndPath(process.o1)
process.XMLIdealGeometryESSource.geomXMLFiles.append('Alignment/LaserAlignmentSimulation/data/AlignmentTubes.xml')
process.g4SimHits.Physics.type = 'SimG4Core/Physics/LaserOpticalPhysics'
process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    NumberOfPhotonsInEachBeam = cms.untracked.int32(100),
    NumberOfPhotonsInParticleGun = cms.untracked.int32(10),
    SiAbsorptionLengthScalingFactor = cms.untracked.double(1.0),
    PhotonEnergy = cms.untracked.double(1.15),
    MaterialPropertiesDebugLevel = cms.untracked.int32(1),
    DebugLevel = cms.untracked.int32(3),
    EnergyLossScalingFactor = cms.untracked.double(1739.130435),
    type = cms.string('LaserAlignmentSimulation')
))


# Auto generated configuration file
# using: 
# Revision: 1.20 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: FourMuPt_1_200_cfi --conditions auto:upgradePLS3 -n 10 --eventcontent FEVTDEBUG --relval 10000,100 -s GEN,SIM --datatier GEN-SIM --beamspot Gauss --customise SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2023NoEE --geometry Extended2023HGCalMuon4Eta,Extended2023HGCalMuon4EtaReco --magField 38T_PostLS1 --fileout file:step1.root
import FWCore.ParameterSet.Config as cms

process = cms.Process('SIM')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
#process.load('Configuration.Geometry.GeometryExtended2023HGCalMuon4EtaReco_cff')
#process.load('Configuration.Geometry.GeometryExtended2023HGCalMuon4Eta_cff')
process.load('Configuration.Geometry.GeometryExtended2023SHCalNoTaper4EtaReco_cff')
process.load('Configuration.Geometry.GeometryExtended2023SHCalNoTaper4Eta_cff')
process.load("Geometry.HGCalCommonData.testShashlikXML_cfi")
process.load("Geometry.HGCalCommonData.shashlikNumberingInitialization_cfi")
process.load("Geometry.CaloEventSetup.ShashlikTopology_cfi")
process.load("Geometry.CaloTopology.ShashlikGeometryESProducer_cfi")

process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedGauss_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.load('RecoParticleFlow.PFClusterProducer.particleFlowRecHitShashlik_cfi')
process.load('RecoParticleFlow.PFClusterProducer.particleFlowClusterShashlik_cfi')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

# Input source
process.source = cms.Source("EmptySource")

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.20 $'),
    annotation = cms.untracked.string('FourMuPt_1_200_cfi nevts:10'),
    name = cms.untracked.string('Applications')
)

# Output definition

process.FEVTDEBUGoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = process.FEVTDEBUGEventContent.outputCommands+['keep *_*_EKSimRecoHits_*',
								   'keep *_particleFlowRecHitShashlik_*_*',
								   'keep *_particleFlowClusterEKUncorrected_*_*'],
    fileName = cms.untracked.string('file:step1.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('GEN-SIM')
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('generation_step')
    )
)

# Additional output definition

# Other statements
process.genstepfilter.triggerConditions=cms.vstring("generation_step")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')

process.generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        MaxE = cms.double(276.0),
        MinE = cms.double(276.0),
        PartID = cms.vint32(11),
        MinEta = cms.double(2.4999),
        MaxEta = cms.double(2.5001),
        MaxPhi = cms.double(3.14159265359),
        MinPhi = cms.double(-3.14159265359)
    ),
    Verbosity = cms.untracked.int32(0),
    psethack = cms.string('Four mu pt 1 to 200'),
    AddAntiParticle = cms.bool(False),
    firstRun = cms.untracked.uint32(1)
)

process.prec = cms.EDProducer("CaloSimhitToRechitProducer",
                              src = cms.InputTag("g4SimHits", "EcalHitsEK"),
			      energyScale = cms.double (1.64)
)

process.particleFlowRecHitShashlik.producers[0].src = cms.InputTag("prec:EKSimRecoHits")

process.pfClusterAnalyzer = cms.EDAnalyzer("PFClusterAnalyzer",
    PFClusters = cms.InputTag("particleFlowClusterEKUncorrected"),
    verbose = cms.untracked.bool(True),
    printBlocks = cms.untracked.bool(False)
)

# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.reco_step = cms.Path(process.prec+process.particleFlowRecHitShashlik+process.particleFlowClusterEKUncorrected)#+process.pfClusterAnalyzer)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGoutput_step = cms.EndPath(process.FEVTDEBUGoutput)

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.simulation_step,process.reco_step,process.endjob_step,process.FEVTDEBUGoutput_step)
# filter all path with the production filter sequence
for path in process.paths:
	getattr(process,path)._seq = process.generator * getattr(process,path)._seq 

# customisation of the process.

# Automatic addition of the customisation function from SLHCUpgradeSimulations.Configuration.combinedCustoms
from SLHCUpgradeSimulations.Configuration.combinedCustoms import cust_2023NoEE 

#call to customisation function cust_2023NoEE imported from SLHCUpgradeSimulations.Configuration.combinedCustoms
process = cust_2023NoEE(process)

# End of customisation functions

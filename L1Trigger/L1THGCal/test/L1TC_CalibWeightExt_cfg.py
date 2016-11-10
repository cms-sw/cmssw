import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

process = cms.Process('test',eras.Phase2C2)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2023D3Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2023D3_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('Configuration.StandardSequences.VtxSmearedNoSmear_cff')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.DigiToRaw_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.load('L1Trigger.L1THGCal.hgcalTriggerPrimitives_cff')
process.load('L1Trigger.L1THGCal.hgcalTriggerGeometryESProducer_cfi')
process.prefer("hgcalTriggerGeometryESProducer") 


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Input source
#process.source = cms.Source("EmptySource")
process.source = cms.Source("PoolSource", 
#                            fileNames = cms.untracked.vstring('file:/afs/cern.ch/work/l/lmastrol/Branch/branchTest/test/HGC_L1Calib_CMSSW/tmp/CMSSW_8_1_0_pre12/src/L1Trigger/L1THGCal/test/SinglePhoton_Pt100_eta2_14tc.root')
                            fileNames = cms.untracked.vstring('file:/afs/cern.ch/work/l/lmastrol/Branch/branchTest/test/HGC_L1Calib_CMSSW/tmp/CMSSW_8_1_0_pre12/src/L1Trigger/L1THGCal/test/Test_NoTruncationE16bits.root')
)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.20 $'),
    annotation = cms.untracked.string('SingleElectronPt10_cfi nevts:10'),
    name = cms.untracked.string('Applications')
)

# Additional output definition
process.TFileService = cms.Service(
    "TFileService",
#    fileName = cms.string("L1tpg_SinglePhotonPt100_12tcXmodule.root")
    fileName = cms.string("Output_NoTruncationE16bits.root")

    )



# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')


#process.mix.digitizers = cms.PSet(process.theDigitizersValid)


# Path and EndPath definitions
process.digi2raw_step = cms.Path(process.DigiToRaw)
process.endjob_step = cms.EndPath(process.endOfProcess)

process.geometryProducer = cms.ESProducer(
    'HGCalTriggerGeometryESProducer',
    TriggerGeometry = cms.PSet(
        TriggerGeometryName = cms.string('HGCalTriggerGeometryHexImp2'),
        L1TCellsMapping = cms.FileInPath("L1Trigger/L1THGCal/data/triggercell_mapping.txt"),
        L1TModulesMapping = cms.FileInPath("L1Trigger/L1THGCal/data/module_mapping.txt"),
        eeSDName = cms.string('HGCalEESensitive'),
        fhSDName = cms.string('HGCalHESiliconSensitive'),
        bhSDName = cms.string('HGCalHEScintillatorSensitive'),
        )
)

process.hgcalTriggerPrimitiveDigiProducer.FECodec.NData = cms.uint32(12)
       
process.TC_CalibWeight = cms.EDAnalyzer("testCalibration",
                                        DebugCode=cms.bool(False)
#                                        FECodec = process.hgcalTriggerPrimitiveDigiProducer.FECodec
                                        #FECodec = process.hgcalTriggerPrimitiveDigiFEReproducer.FECodec.clone()

                                            # GenParticleCollection=cms.InputTag("genParticles"),
                                            # HydraTag=cms.InputTag("Hydra"),
                                            # SplitRecHits=cms.bool(False),
                                            # UseGenParticles=cms.bool(False),
                                            # MinDebugEnergy=cms.untracked.double(0.)
                                        )



#process.hgcaltriggergeomtester = cms.EDAnalyzer(
#    "HGCalTriggerGeomTester"
#    )


#process.test_step = cms.Path(process.hgcaltriggergeomtester)
process.test_step = cms.Path(process.TC_CalibWeight)



# Schedule definition
#process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.simulation_step,process.digitisation_step,process.L1simulation_step,process.digi2raw_step,process.test_step,process.endjob_step,process.FEVTDEBUGoutput_step)
process.schedule = cms.Schedule(process.test_step,process.endjob_step)
#process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.simulation_step,process.digitisation_step,process.L1simulation_step,process.digi2raw_step,process.endjob_step,process.FEVTDEBUGoutput_step)
# filter all path with the production filter sequence
for path in process.paths:
    getattr(process,path)._seq = getattr(process,path)._seq

# customisation of the process.

# Automatic addition of the customisation function from SLHCUpgradeSimulations.Configuration.combinedCustoms
from SLHCUpgradeSimulations.Configuration.combinedCustoms import cust_2023tilted

#call to customisation function cust_2023tilted imported from SLHCUpgradeSimulations.Configuration.combinedCustoms
process = cust_2023tilted(process)



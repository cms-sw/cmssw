# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step2 --datamix PreMix --conditions auto:run2_mc --pileup_input das:/RelValPREMIXUP15_PU25/CMSSW_7_4_0_pre6-PU25ns_MCRUN2_74_V1-v1/GEN-SIM-DIGI-RAW --eventcontent FEVTDEBUGHLT -s DIGIPREMIX_S2:pdigi_valid,DATAMIX,L1,DIGI2RAW,HLT:@relval,RAW2DIGI,L1Reco --datatier GEN-SIM-DIGI-RAW-HLTDEBUG --customise SLHCUpgradeSimulations/Configuration/postLS1CustomsPreMixing.customisePostLS1 --magField 38T_PostLS1 -n 100 --no_exec --python_filename premix_step2_full_cfg.py
import FWCore.ParameterSet.Config as cms

process = cms.Process('HLT')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('FastSimulation.Configuration.EventContent_cff') #!!
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('FastSimulation.Configuration.Geometries_MC_cff') #!!
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load('Configuration.StandardSequences.Generator_cff') #!!
process.load('IOMC.EventVertexGenerators.VtxSmearedNominalCollision2015_cfi') #!!
process.load('GeneratorInterface.Core.genFilterSummary_cff') #!!
process.load('FastSimulation.Configuration.SimIdeal_cff') #!!
process.load("FastSimulation.Configuration.Reconstruction_Tk_cff") #!!
process.load('FastSimulation.Configuration.DigiDMPreMix_cff') #!!
process.load('SimGeneral.MixingModule.digi_MixPreMix_cfi')
process.load('FastSimulation.Configuration.DataMixerPreMix_cff') #!!
process.load('FastSimulation.Configuration.SimL1EmulatorDM_cff') #!!
process.load('FastSimulation.Configuration.DigiToRawDM_cff') #!!
process.load('HLTrigger.Configuration.HLT_GRun_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load("FastSimulation.Configuration.Reconstruction_NoTk_cff") #!!
#process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

# Input source
process.source = cms.Source("EmptySource") #!!!

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('step2 nevts:100'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.FEVTDEBUGHLToutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-DIGI-RAW-HLTDEBUG'),
        filterName = cms.untracked.string('')
    ),
    eventAutoFlushCompressedSize = cms.untracked.int32(1048576),
    fileName = cms.untracked.string('step2_DIGIPREMIX_S2_DATAMIX_L1_DIGI2RAW_HLT_RAW2DIGI_L1Reco.root'),
    outputCommands = process.FEVTDEBUGHLTEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition

# Other statements
process.mix.digitizers = cms.PSet(process.theDigitizersMixPreMixValid)
process.mixData.input.fileNames = cms.untracked.vstring(['file:minbias_premixed.root'])
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

process.generator = cms.EDFilter("Pythia8GeneratorFilter",
    PythiaParameters = cms.PSet(
        parameterSets = cms.vstring('pythia8CommonSettings',
            'pythia8CUEP8M1Settings',
            'processParameters'),
        processParameters = cms.vstring('Top:gg2ttbar = on ',
            'Top:gg2ttbar = on ',
            '6:m0 = 175 '),
        pythia8CUEP8M1Settings = cms.vstring('Tune:pp 14',
            'Tune:ee 7',
            'MultipartonInteractions:pT0Ref=2.4024',
            'MultipartonInteractions:ecmPow=0.25208',
            'MultipartonInteractions:expPow=1.6'),
        pythia8CommonSettings = cms.vstring('Tune:preferLHAPDF = 2',
            'Main:timesAllowErrors = 10000',
            'Check:epTolErr = 0.01',
            'Beams:setProductionScalesFromLHEF = off',
            'SLHA:keepSM = on',
            'SLHA:minMassSM = 1000.',
            'ParticleDecays:limitTau0 = on',
            'ParticleDecays:tau0Max = 10',
            'ParticleDecays:allowPhotonRadiation = on')
    ),
    comEnergy = cms.double(13000.0),
    filterEfficiency = cms.untracked.double(1.0),
    maxEventsToPrint = cms.untracked.int32(0),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    pythiaPylistVerbosity = cms.untracked.int32(0)
)

process.ProductionFilterSequence = cms.Sequence(process.generator)

# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.reconstruction_tk_step = cms.Path(process.fastTkReconstruction)
process.digitisation_step = cms.Path(process.pdigi_valid)
process.datamixing_step = cms.Path(process.pdatamix)
process.L1simulation_step = cms.Path(process.SimL1Emulator)
process.digi2raw_step = cms.Path(process.DigiToRaw)
process.raw2digi_step = cms.Path(process.RawToDigi)
process.reconstruction_notk_step = cms.Path(process.reconstruction) #!!
#process.L1Reco_step = cms.Path(process.L1Reco) #!!
process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGHLToutput_step = cms.EndPath(process.FEVTDEBUGHLToutput)

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,process.simulation_step,process.reconstruction_tk_step,process.digitisation_step,process.datamixing_step,process.L1simulation_step,process.digi2raw_step,process.raw2digi_step,process.reconstruction_notk_step,process.endjob_step,process.FEVTDEBUGHLToutput_step)
# filter all path with the production filter sequence
for path in process.paths:
	getattr(process,path)._seq = process.ProductionFilterSequence * getattr(process,path)._seq 

# customisation of the process.

# Automatic addition of the customisation function from FastSimulation.Configuration.MixingModule_Full2Fast
from FastSimulation.Configuration.MixingModule_Full2Fast import prepareDigiRecoMixing 

#call to customisation function prepareDigiRecoMixing imported from FastSimulation.Configuration.MixingModule_Full2Fast
process = prepareDigiRecoMixing(process)

# Automatic addition of the customisation function from SLHCUpgradeSimulations.Configuration.postLS1CustomsPreMixing
from SLHCUpgradeSimulations.Configuration.postLS1CustomsPreMixing import customisePostLS1 

#call to customisation function customisePostLS1 imported from SLHCUpgradeSimulations.Configuration.postLS1CustomsPreMixing
process = customisePostLS1(process)

# Automatic addition of the customisation function from HLTrigger.Configuration.customizeHLTforMC
from HLTrigger.Configuration.customizeHLTforMC import customizeHLTforMC 

#call to customisation function customizeHLTforMC imported from HLTrigger.Configuration.customizeHLTforMC
process = customizeHLTforMC(process)

# End of customisation functions


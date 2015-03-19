import FWCore.ParameterSet.Config as cms

process = cms.Process("DIGI")

process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('FastSimulation.Configuration.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('FastSimulation.Configuration.Geometries_MC_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')

process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedNominalCollision2015_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load("FastSimulation/Configuration/SimIdeal_cff")
process.load("FastSimulation/Configuration/Reconstruction_BefMix_cff")
process.load('FastSimulation.Configuration.Digi_cff')
#process.load('FastSimulation.Configuration.SimL1Emulator_cff')
#process.load('FastSimulation.Configuration.DigiToRaw_cff')
#process.load('FastSimulation.Configuration.RawToDigi_cff')
process.load("FastSimulation/Configuration/Reconstruction_AftMix_cff")
process.load('CommonTools.ParticleFlow.EITopPAG_cff')
process.load('HLTrigger.Configuration.HLT_GRun_Famos_cff')

process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source("EmptySource")

process.FEVTDEBUGoutput = cms.OutputModule("PoolOutputModule",
    eventAutoFlushCompressedSize = cms.untracked.int32(1048576),
    fileName = cms.untracked.string('final.root'),
    outputCommands = process.FEVTDEBUGEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

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

process.generation_step = cms.Path(process.pgen)
process.simstep = cms.Path(process.psim)
process.step0 = cms.Path(process.reconstruction_befmix)
process.step = cms.Path(process.pdigi)
#process.step2 = cms.Path(process.SimL1Emulator)
#process.step3 = cms.Path(process.DigiToRaw )
#process.step4 = cms.Path(process.RawToDigi )
process.step5 = cms.Path(process.reconstruction )
process.eventinterpretaion_step = cms.Path(process.EIsequence)
process.HLTEndSequence = cms.Sequence( process.dummyModule )

process.FEVTDEBUGoutput_step = cms.EndPath(process.FEVTDEBUGoutput)

#process.schedule = cms.Schedule(process.generation_step,process.simstep,process.step0,process.step,process.step2,process.step3,process.step4,process.step5,process.eventinterpretaion_step,process.HLTSchedule,process.FEVTDEBUGoutput_step)
process.schedule = cms.Schedule(process.generation_step,process.simstep,process.step0,process.step,process.step5,process.FEVTDEBUGoutput_step)
# Automatic addition of the customisation function from FastSimulation.Configuration.MixingModule_Full2Fast

for path in process.paths:
        getattr(process,path)._seq = process.ProductionFilterSequence * getattr(process,path)._seq

from FastSimulation.Configuration.MixingModule_Full2Fast import prepareDigiRecoMixing 

#call to customisation function prepareDigiRecoMixing imported from FastSimulation.Configuration.MixingModule_Full2Fast
process = prepareDigiRecoMixing(process)

# Automatic addition of the customisation function from SLHCUpgradeSimulations.Configuration.postLS1Customs
from SLHCUpgradeSimulations.Configuration.postLS1Customs import customisePostLS1 

#call to customisation function customisePostLS1 imported from SLHCUpgradeSimulations.Configuration.postLS1Customs
process = customisePostLS1(process)

# Automatic addition of the customisation function from HLTrigger.Configuration.customizeHLTforMC                                                                                                           
from HLTrigger.Configuration.customizeHLTforMC import customizeHLTforMC

#call to customisation function customizeHLTforMC imported from HLTrigger.Configuration.customizeHLTforMC                                                                                                   
process = customizeHLTforMC(process)

# End of customisation functions

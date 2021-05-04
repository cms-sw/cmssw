#Test of complete gen flow for an specific BPH signal decay.
import FWCore.ParameterSet.Config as cms

process = cms.Process('GEN')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedRealistic50ns13TeVCollision_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(200)
)

# Input source
process.source = cms.Source("EmptySource")

process.options = cms.untracked.PSet()

# Output definition

process.RAWSIMoutput = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('generation_step')
    ),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN'),
        filterName = cms.untracked.string('')
    ),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    fileName = cms.untracked.string('file:gen_lambdab.root'),
    outputCommands = process.RAWSIMEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition

# Other statements
process.genstepfilter.triggerConditions=cms.vstring("generation_step")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

process.generator = cms.EDFilter("Pythia8GeneratorFilter",
    ExternalDecays = cms.PSet(
        EvtGen1 = cms.untracked.PSet(
            convertPythiaCodes = cms.untracked.bool(False),
            decay_table = cms.string('GeneratorInterface/EvtGenInterface/data/DECAY_NOLONGLIFE.DEC'),
            list_forced_decays = cms.vstring('MyLambda_b0','Myanti-Lambda_b0'),
            particle_property_file = cms.FileInPath('GeneratorInterface/EvtGenInterface/data/evt.pdl'),
	    operates_on_particles = cms.vint32(5122),  #only care about signal particle
            user_decay_embedded = cms.vstring(
"""
Alias      MyLambda_b0   Lambda_b0
Alias      Myanti-Lambda_b0   anti-Lambda_b0
ChargeConj Myanti-Lambda_b0   MyLambda_b0
#
Decay MyLambda_b0
1.000      p+  mu-   anti-nu_mu    PHOTOS Lb2plnuLCSR 1 1 1 1;
#1.000      p+  mu-   anti-nu_mu   Lb2plnuLCSR 1 1 1 1;
Enddecay
CDecay Myanti-Lambda_b0
#
End
"""
            )
        ),
        parameterSets = cms.vstring('EvtGen1')
    ),
    PythiaParameters = cms.PSet(
        parameterSets = cms.vstring('pythia8CommonSettings', 
            'pythia8CUEP8M1Settings', 
            'processParameters'),
        processParameters = cms.vstring(
	    'SoftQCD:nonDiffractive = on', 
	    'PTFilter:filter = on',
	    'PTFilter:quarkToFilter = 5',
            'PTFilter:scaleToFilter = 1.0'),
        pythia8CUEP8M1Settings = cms.vstring('Tune:pp 14', 
            'Tune:ee 7', 
            'MultipartonInteractions:pT0Ref=2.4024', 
            'MultipartonInteractions:ecmPow=0.25208', 
            'MultipartonInteractions:expPow=1.6'),
        pythia8CommonSettings = cms.vstring('Tune:preferLHAPDF = 2', 
            'Main:timesAllowErrors = 10000', 
            'Check:epTolErr = 0.01', 
            'Beams:setProductionScalesFromLHEF = off', 
            'SLHA:minMassSM = 1000.', 
            'ParticleDecays:limitTau0 = on', 
            'ParticleDecays:tau0Max = 10', 
            'ParticleDecays:allowPhotonRadiation = on')
    ),
    comEnergy = cms.double(13000.0),
    maxEventsToPrint = cms.untracked.int32(0),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    pythiaPylistVerbosity = cms.untracked.int32(0)
)

# The line below removes messages like "particle not recognized by pythia"
from GeneratorInterface.EvtGenInterface.EvtGenSetting_cff import *
process.generator.PythiaParameters.processParameters.extend(EvtGenExtraParticles)

process.lbfilter = cms.EDFilter("PythiaDauVFilter",
    DaughterIDs = cms.untracked.vint32(2212, 13),
    MaxEta = cms.untracked.vdouble(2.5, 2.5),
    MinEta = cms.untracked.vdouble(-2.5, -2.5),
    MinPt = cms.untracked.vdouble(0.4, 1.5),
    MotherID = cms.untracked.int32(0),
    NumberDaughters = cms.untracked.int32(2),
    ParticleID = cms.untracked.int32(5122)
)

process.ProductionFilterSequence = cms.Sequence(process.generator+process.lbfilter)

# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.RAWSIMoutput_step = cms.EndPath(process.RAWSIMoutput)

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.endjob_step,process.RAWSIMoutput_step)
# filter all path with the production filter sequence
for path in process.paths:
	getattr(process,path)._seq = process.ProductionFilterSequence * getattr(process,path)._seq 

process.MessageLogger.cerr.FwkReport.reportEvery = 10

# Customisation from command line

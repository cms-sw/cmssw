import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

process = cms.Process('GEN2',eras.Run2_2017)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedRealistic25ns13TeVEarly2017Collision_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

#----------
# Input source
#----------
process.source = cms.Source("LHESource",
    # first few events from /eos/cms/store/lhe/5542/GJets_HT-200To400_8TeV-madgraph_10001.lhe
    fileNames = cms.untracked.vstring("file:GJets_HT-200To400_8TeV-madgraph.lhe")
)

process.options = cms.untracked.PSet()

#----------
# Output definition
#----------
process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('file:double-em-enrichment-test.root'),
)

process.output_step = cms.EndPath(process.output)

#----------
# PYTHIA hadronizer
# with double EM enrichment filter
#----------
from Configuration.Generator.doubleEMEnrichingHepMCfilter_cfi import *
from Configuration.Generator.Pythia8CommonSettings_cfi import *
from Configuration.Generator.MCTunes2017.PythiaCP5Settings_cfi import *

process.generator = cms.EDFilter("Pythia8HadronizerFilter",

    HepMCFilter = cms.PSet(
        filterName = cms.string('PythiaHepMCFilterGammaGamma'),

        # double EM enrichment filter parameters
        filterParameters = doubleEMenrichingHepMCfilterParams
    ),
				 
    PythiaParameters = cms.PSet(
        pythia8CommonSettingsBlock,
        pythia8CP5SettingsBlock,
   
        parameterSets = cms.vstring('pythia8CommonSettings', 
             'pythia8CP5Settings', 
             'processParameters'),

        # e.g. from /GJets_HT-40To100_TuneCP5_13TeV-madgraphMLM-pythia8 configuration
        processParameters = cms.vstring('JetMatching:setMad = off', 
            'JetMatching:scheme = 1', 
            'JetMatching:merge = on', 
            'JetMatching:jetAlgorithm = 2', 
            'JetMatching:etaJetMax = 5.', 
            'JetMatching:coneRadius = 1.', 
            'JetMatching:slowJetPower = 1', 
            'JetMatching:qCut = 19.', 
            'JetMatching:nQmatch = 5', 
            'JetMatching:nJetMax = 4', 
            'JetMatching:doShowerKt = off'),
        ),

        comEnergy = cms.double(13000.0),
        filterEfficiency = cms.untracked.double(1.0),
        maxEventsToPrint = cms.untracked.int32(0),
        
        # number of hadronization attempts per LHE event
        nAttempts = cms.uint32(20), 
        
        pythiaHepMCVerbosity = cms.untracked.bool(False),
        pythiaPylistVerbosity = cms.untracked.int32(1),
)


# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.endjob_step = cms.EndPath(process.endOfProcess)

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,
                                process.genfiltersummary_step,
                                process.endjob_step,
                                process.output_step,
)

from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)

for path in process.paths:
    if path in ['lhe_step']: continue
    getattr(process,path)._seq = process.generator * getattr(process,path)._seq 

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100))

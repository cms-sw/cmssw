# Auto generated configuration file
# using: 
# Revision: 1.400 
# Source: /local/reps/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: WtoMuNu_14TeV_cfi.py -s GEN,SIM --conditions POSTLS161_V12::All --geometry Geometry/GEMGeometry/cmsExtendedGeometryPostLS1plusGEMXML_cfi --datatier GEN-SIM --eventcontent FEVTDEBUG -n 200 --no_exec --fileout out_sim.root --evt_type WtoMuNu_14TeV_cfi.py
import FWCore.ParameterSet.Config as cms

process = cms.Process('SIM')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Geometry.GEMGeometry.cmsExtendedGeometryPostLS1plusGEMXML_cfi')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedRealistic8TeVCollision_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000)
)

# Input source
process.source = cms.Source("EmptySource")

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.400 $'),
    annotation = cms.untracked.string('WtoMuNu_14TeV_cfi.py nevts:200'),
    name = cms.untracked.string('PyReleaseValidation')
)

# Output definition

process.FEVTDEBUGoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = process.FEVTDEBUGEventContent.outputCommands,
    fileName = cms.untracked.string('out_sim.root'),
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
process.GlobalTag = GlobalTag(process.GlobalTag, 'POSTLS161_V12::All', '')

from Configuration.Generator.PythiaUEZ2starSettings_cfi import *
process.generator = cms.EDFilter("Pythia6GeneratorFilter",
    pythiaPylistVerbosity = cms.untracked.int32(0),
    filterEfficiency = cms.untracked.double(1.0),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    crossSection = cms.untracked.double(1774.0),
    comEnergy = cms.double(14000.0),
    maxEventsToPrint = cms.untracked.int32(0),
    PythiaParameters = cms.PSet(
        pythiaUESettingsBlock,
        processParameters = cms.vstring(
            'MSEL        = 0    !User defined processes',
            'MSUB(2)     = 1    !W production',
            'MDME(190,1) = 0    !W decay into dbar u',
            'MDME(191,1) = 0    !W decay into dbar c',
            'MDME(192,1) = 0    !W decay into dbar t',
            'MDME(194,1) = 0    !W decay into sbar u',
            'MDME(195,1) = 0    !W decay into sbar c',
            'MDME(196,1) = 0    !W decay into sbar t',
            'MDME(198,1) = 0    !W decay into bbar u',
            'MDME(199,1) = 0    !W decay into bbar c',
            'MDME(200,1) = 0    !W decay into bbar t',
            'MDME(206,1) = 0    !W decay into e+ nu_e',
            'MDME(207,1) = 1    !W decay into mu+ nu_mu',
            'MDME(208,1) = 0    !W decay into tau+ nu_tau'),
        # This is a vector of ParameterSet names to be read, in this order
        parameterSets = cms.vstring('pythiaUESettings',
                                    'processParameters'
                                    ),
    )
)

# select generated muons and antimuons
process.genMuons = cms.EDFilter("PdgIdCandViewSelector",
    src = cms.InputTag("genParticles"),
    pdgId = cms.vint32( 13, -13 )
)

# filter by applying cuts to these generated muons
process.genMuonsGEM = cms.EDFilter("CandViewSelector",
    src = cms.InputTag("genMuons"),
    cut = cms.string( "abs(eta)<2.4 & abs(eta)>0.9" ), 
    filter = cms.bool(True)
)

process.gen_mu_select = cms.Sequence(process.genMuons * process.genMuonsGEM)

# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen * process.gen_mu_select)
process.simulation_step = cms.Path(process.gen_mu_select * process.psim) 
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGoutput_step = cms.EndPath(process.FEVTDEBUGoutput)

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.simulation_step,process.endjob_step,process.FEVTDEBUGoutput_step)

for path in process.paths:
            getattr(process,path)._seq = process.generator * getattr(process,path)._seq
            

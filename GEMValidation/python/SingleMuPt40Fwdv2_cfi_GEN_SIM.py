# Auto generated configuration file
# using: 
# Revision: 1.400 
# Source: /local/reps/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: SingleMuPt40Fwdv2_cfi -s GEN,SIM --conditions POSTLS161_V12::All --geometry Geometry/GEMGeometry/cmsExtendedGeometryPostLS1plusGEMXML_cfi --datatier GEN-SIM --eventcontent FEVTDEBUG --evt_type RPCGEM/GEMValidation/SingleMuPt40Fwdv2_cfi -n 200 --no_exec --fileout out_sim.root
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
    input = cms.untracked.int32(200)
)

# Input source
process.source = cms.Source("EmptySource")

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.400 $'),
    annotation = cms.untracked.string('RPCGEM/GEMValidation/SingleMuPt40Fwdv2_cfi nevts:200'),
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

## More random muon gun
# select generated muons and antimuons
process.genMuons = cms.EDFilter("PdgIdCandViewSelector",
                                src = cms.InputTag("genParticles"),
                                pdgId = cms.vint32( 13, -13 )
                                )

# filter by applying cuts to these generated muons
process.genMuonsGEM = cms.EDFilter("CandViewSelector",
                                   src = cms.InputTag("genMuons"),
                                   cut = cms.string( "pt > 15. & abs(eta)<2.2 & abs(eta)>1.5" ),   #  or whatever cut expression is deemed necessary
                                   filter = cms.bool(True)
                                   )

process.gen_mu_select = cms.Sequence(process.genMuons * process.genMuonsGEM * process.twoGenMuonsGEM)

# Path and EndPath definitions
#process.generation_step = cms.Path(process.pgen)
process.generation_step = cms.Path(process.pgen * process.gen_mu_select)
process.simulation_step = cms.Path(process.psim)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGoutput_step = cms.EndPath(process.FEVTDEBUGoutput)

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.simulation_step,process.endjob_step,process.FEVTDEBUGoutput_step)

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

# Auto generated configuration file
# using: 
# Revision: 1.400 
# Source: /local/reps/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: SingleMuPt100_cfi -s GEN,SIM --conditions POSTLS161_V12::All --geometry Geometry/GEMGeometry/cmsExtendedGeometryPostLS1plusGEMXML_cfi --datatier GEN-SIM --eventcontent FEVTDEBUG -n 200 --no_exec --fileout out_sim.root
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
    input = cms.untracked.int32(2)
)

# Input source
process.source = cms.Source("EmptySource")

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.400 $'),
    annotation = cms.untracked.string('SingleMuPt100_cfi nevts:200'),
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

process.generator = cms.EDProducer("FlatRandomPtGunProducer",
    PGunParameters = cms.PSet(
        MaxPt = cms.double(40.01),
        MinPt = cms.double(39.99),
        PartID = cms.vint32(-13),
        MaxEta = cms.double(2.2),
        MaxPhi = cms.double(3.14159265359),
        MinEta = cms.double(-2.2),
        MinPhi = cms.double(-3.14159265359)
    ),
    Verbosity = cms.untracked.int32(0),
    psethack = cms.string('single mu pt 40 forward'),
    AddAntiParticle = cms.bool(True),
    firstRun = cms.untracked.uint32(1)
)

# select generated muons and antimuons
process.genMuons = cms.EDFilter("PdgIdCandViewSelector",
 src = cms.InputTag("genParticles"),
 pdgId = cms.vint32( 13, -13 )
)

# filter by applying cuts to these generated muons
process.genMuonsGEM = cms.EDFilter("CandViewSelector",
   src = cms.InputTag("genMuons"),
   cut = cms.string( "abs(eta)<2.14 & abs(eta)>1.45 & phi>0.08 & phi<0.44" ),   #  or whatever cut expression is deemed necessary
   filter = cms.bool(True)
)

#process.genANDfilter = cms.Sequence(process.generator * process.genMuons * process.genMuonsGEM)

process.gen_mu_select = cms.Sequence(process.genMuons * process.genMuonsGEM)

# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen * process.gen_mu_select)
process.simulation_step = cms.Path(process.psim)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGoutput_step = cms.EndPath(process.FEVTDEBUGoutput)

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.simulation_step,process.endjob_step,process.FEVTDEBUGoutput_step)
# filter all path with the production filter sequence
for path in process.paths:
        getattr(process,path)._seq = process.generator * getattr(process,path)._seq
#for path in process.paths:
#	getattr(process,path)._seq = process.genANDfilter * getattr(process,path)._seq 


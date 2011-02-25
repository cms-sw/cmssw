import FWCore.ParameterSet.Config as cms

process = cms.Process('SIM')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.MixingNoPileUp_cff')
process.load('Configuration.StandardSequences.GeometryExtended_cff')
process.load('Configuration.StandardSequences.MagneticField_0T_cff')
process.load('Configuration.StandardSequences.Generator_cff')

process.load('Configuration.StandardSequences.VtxSmearedEarly7TeVCollision_cff')

process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.EventContent.EventContent_cff')

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import *
process.VtxSmeared = cms.EDProducer("BetafuncEvtVtxGenerator",
                                       VtxSmearedCommon,
                                       Phi = cms.double(0.0),
                                       BetaStar = cms.double(1100.0),
                                       Emittance = cms.double(1.0e-07),
                                       Alpha = cms.double(0.0),
                                       SigmaZ = cms.double(2.22),
                                       TimeOffset = cms.double(0.0),

                                  X0 = cms.double(0.2417), # 0.09419   + 0.14750
                                  Y0 = cms.double(0.3855), # 0.007286  + 0.3782
                                  Z0 = cms.double(0.8685) # 0.3838    + 0.4847
                                  )

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    annotation = cms.untracked.string('PYTHIA6-MinBias at 900GeV'),
    name = cms.untracked.string('$Source: /cvs_server/repositories/CMSSW/CMSSW/GeneratorInterface/PartonShowerVeto/Test_PSVInterface.py,v $')
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound')
)
# Input source
process.source = cms.Source("LHESource",
    fileNames = cms.untracked.vstring(
#    'file:zjets_short.lhe')           # Typical lhe file for KtMLM use
    'file:zjets_short_pdfweight.lhe')  # lhe file made with pdf reweighting->suitable for ShowerKt
#    'file:sqsq_fullstat.lhe')          # Typical SUSY lhe file (squark-squark production): test for resonance exclusion (here gluino since SPS1a is used) 
)


# Output definition
process.output = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    outputCommands = cms.untracked.vstring('drop *','keep recoGenJets*_*_*_*'),
    fileName = cms.untracked.string("edmfile.root"),
    dataset = cms.untracked.PSet(
         dataTier = cms.untracked.string('GEN-SIM-RAW'),
         filterName = cms.untracked.string('')
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('generation_step')
    )
)

process.GlobalTag.globaltag = 'START3X_V25B::All'
from Configuration.Generator.PythiaUESettings_cfi import *

process.RandomNumberGeneratorService.generator.initialSeed = cms.untracked.uint32(1801)

process.generator = cms.EDFilter("Pythia6HadronizerFilter",
     pythiaHepMCVerbosity = cms.untracked.bool(False),
     maxEventsToPrint = cms.untracked.int32(10),
     pythiaPylistVerbosity = cms.untracked.int32(2),
     comEnergy = cms.double(7000.0),
     PythiaParameters = cms.PSet(
         pythiaUESettingsBlock,
         processParameters = cms.vstring('MSEL=0         ! User defined processes',

                         'PMAS(5,1)=4.4   ! b quark mass',
                         'PMAS(6,1)=172.4 ! t quark mass',
                         'MSTJ(1)=1       ! Fragmentation/hadronization on or off',
                         'MSTP(61)=1      ! Parton showering on or off',
                         'MSTP(81)=20     ! Use pt-ordered showers',
                        'MSTP(143)=1     ! Starting scale of showers',


		), 
         # This is a vector of ParameterSet names to be read, in this order
         parameterSets = cms.vstring('pythiaUESettings', 
             'processParameters')
     ),
     jetMatching = cms.untracked.PSet(
        scheme = cms.string("Madgraph"),
        mode = cms.string("auto"),         # soup, or "inclusive" / "exclusive"
        MEMAIN_etaclmax = cms.double(5.0),

######### Matching scale: qcut>xqcut for KtMLM, qcut=xqcut for ShowerKt  
        MEMAIN_qcut = cms.double(15.0), #ShowerKt (since xqcut=15 and pdfwgt=T)
#       MEMAIN_qcut = cms.double(15.0), #KtMLM 

####### Use ShowerKt: 1=yes (only with pt-ordered showers!)
#	MEMAIN_showerkt = cms.double(0),
	MEMAIN_showerkt = cms.double(1),   

        MEMAIN_minjets = cms.int32(0),     # min number of ISR partons in the lhe file
        MEMAIN_maxjets = cms.int32(2),     # max number of ISR partons in the file
        MEMAIN_excres = cms.string(""),    # write the resonances PID to exclude, e.g."1000021,1000001" 
        outTree_flag = cms.int32(1)        # Decides if the .tree file must be written for further sanity check (DJR plots)
     )

)
process.ProductionFilterSequence = cms.Sequence(process.generator)

# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen)
process.endjob_step = cms.Path(process.endOfProcess)
process.out_step = cms.EndPath(process.output)

# Schedule definition
process.schedule = cms.Schedule(process.generation_step)
process.schedule.extend([process.endjob_step,process.out_step])
# special treatment in case of production filter sequence
for path in process.paths:
    getattr(process,path)._seq = process.ProductionFilterSequence*getattr(process,path)._seq


process.MessageLogger.cerr.threshold = 'DEBUG'

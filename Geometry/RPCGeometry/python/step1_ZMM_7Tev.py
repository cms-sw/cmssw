# Auto generated configuration file
# using: 
# Revision: 1.173 
# Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: ZMM_7TeV.cfi -s GEN:ProductionFilterSequence,SIM,DIGI,L1,DIGI2RAW,HLT,RAW2DIGI,L1Reco -n 10 --geometry DB --conditions auto:startup --customise Geometry/RPCGeometry/customise_RPCgeom37X.py --relval 18000,200 --datatier GEN-SIM-DIGI-RAW-HLTDEBUG --eventcontent FEVTDEBUGHLT --fileout file:raw.root
import FWCore.ParameterSet.Config as cms

process = cms.Process('HLT')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.MixingNoPileUp_cff')
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('Configuration.StandardSequences.VtxSmearedEarly10TeVCollision_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.DigiToRaw_cff')
process.load('HLTrigger.Configuration.HLT_8E29_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.EventContent.EventContent_cff')

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    annotation = cms.untracked.string('ZMM_7TeV.cfi nevts:10'),
    name = cms.untracked.string('PyReleaseValidation')
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
process.options = cms.untracked.PSet(

)
# Input source
process.source = cms.Source("EmptySource")

# Output definition
process.output = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    outputCommands = process.FEVTDEBUGHLTEventContent.outputCommands,
    fileName = cms.untracked.string('file:raw.root'),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-DIGI-RAW-HLTDEBUG'),
        filterName = cms.untracked.string('')
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('generation_step')
    )
)

# Additional output definition

# Other statements
process.GlobalTag.globaltag = 'START37_V1::All'
process.generator = cms.EDFilter("Pythia6GeneratorFilter",
    pythiaPylistVerbosity = cms.untracked.int32(0),
    filterEfficiency = cms.untracked.double(1.0),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    comEnergy = cms.double(7000.0),
    maxEventsToPrint = cms.untracked.int32(0),
    PythiaParameters = cms.PSet(
        pythiaUESettings = cms.vstring('MSTJ(11)=3     ! Choice of the fragmentation function', 
            'MSTJ(22)=2     ! Decay those unstable particles', 
            'PARJ(71)=10 .  ! for which ctau  10 mm', 
            'MSTP(2)=1      ! which order running alphaS', 
            'MSTP(33)=0     ! no K factors in hard cross sections', 
            'MSTP(51)=10042 ! structure function chosen (external PDF CTEQ6L1)', 
            'MSTP(52)=2     ! work with LHAPDF', 
            'MSTP(81)=1     ! multiple parton interactions 1 is Pythia default', 
            'MSTP(82)=4     ! Defines the multi-parton model', 
            'MSTU(21)=1     ! Check on possible errors during program execution', 
            'PARP(82)=1.8387   ! pt cutoff for multiparton interactions', 
            'PARP(89)=1960. ! sqrts for which PARP82 is set', 
            'PARP(83)=0.5   ! Multiple interactions: matter distrbn parameter', 
            'PARP(84)=0.4   ! Multiple interactions: matter distribution parameter', 
            'PARP(90)=0.16  ! Multiple interactions: rescaling power', 
            'PARP(67)=2.5    ! amount of initial-state radiation', 
            'PARP(85)=1.0  ! gluon prod. mechanism in MI', 
            'PARP(86)=1.0  ! gluon prod. mechanism in MI', 
            'PARP(62)=1.25   ! ', 
            'PARP(64)=0.2    ! ', 
            'MSTP(91)=1      !', 
            'PARP(91)=2.1   ! kt distribution', 
            'PARP(93)=15.0  ! '),
        processParameters = cms.vstring('MSEL         = 11 ', 
            'MDME( 174,1) = 0    !Z decay into d dbar', 
            'MDME( 175,1) = 0    !Z decay into u ubar', 
            'MDME( 176,1) = 0    !Z decay into s sbar', 
            'MDME( 177,1) = 0    !Z decay into c cbar', 
            'MDME( 178,1) = 0    !Z decay into b bbar', 
            'MDME( 179,1) = 0    !Z decay into t tbar', 
            'MDME( 182,1) = 0    !Z decay into e- e+', 
            'MDME( 183,1) = 0    !Z decay into nu_e nu_ebar', 
            'MDME( 184,1) = 1    !Z decay into mu- mu+', 
            'MDME( 185,1) = 0    !Z decay into nu_mu nu_mubar', 
            'MDME( 186,1) = 0    !Z decay into tau- tau+', 
            'MDME( 187,1) = 0    !Z decay into nu_tau nu_taubar', 
            'CKIN( 1)     = 40.  !(D=2. GeV)', 
            'CKIN( 2)     = -1.  !(D=-1. GeV)'),
        parameterSets = cms.vstring('pythiaUESettings', 
            'processParameters')
    )
)
process.mumugenfilter = cms.EDFilter("MCParticlePairFilter",
    Status = cms.untracked.vint32(1, 1),
    MinPt = cms.untracked.vdouble(2.5, 2.5),
    MaxEta = cms.untracked.vdouble(2.5, 2.5),
    MinEta = cms.untracked.vdouble(-2.5, -2.5),
    ParticleCharge = cms.untracked.int32(-1),
    ParticleID1 = cms.untracked.vint32(13),
    ParticleID2 = cms.untracked.vint32(13)
)
process.ProductionFilterSequence = cms.Sequence(process.generator*process.mumugenfilter)

# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.digitisation_step = cms.Path(process.pdigi)
process.L1simulation_step = cms.Path(process.SimL1Emulator)
process.digi2raw_step = cms.Path(process.DigiToRaw)
process.raw2digi_step = cms.Path(process.RawToDigi)
process.L1Reco_step = cms.Path(process.L1Reco)
process.endjob_step = cms.Path(process.endOfProcess)
process.out_step = cms.EndPath(process.output)

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,process.simulation_step,process.digitisation_step,process.L1simulation_step,process.digi2raw_step)
process.schedule.extend(process.HLTSchedule)
process.schedule.extend([process.raw2digi_step,process.L1Reco_step,process.endjob_step,process.out_step])
# special treatment in case of production filter sequence  
for path in process.paths: 
    getattr(process,path)._seq = process.ProductionFilterSequence*getattr(process,path)._seq


# Automatic addition of the customisation function
def customise(process):
    process.GlobalTag.toGet = cms.VPSet(
        cms.PSet(record = cms.string("PCastorRcd"),
                 tag = cms.string("CASTORRECO_Geometry_Tag38YV0"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_GEOMETRY")
                 ),
        cms.PSet(record = cms.string("PZdcRcd"),
                 tag = cms.string("ZDCRECO_Geometry_Tag38YV0"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_GEOMETRY")
                 ),
        cms.PSet(record = cms.string("PCaloTowerRcd"),
                 tag = cms.string("CTRECO_Geometry_Tag38YV0"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_GEOMETRY")
                 ),
        cms.PSet(record = cms.string("PEcalEndcapRcd"),
                 tag = cms.string("EERECO_Geometry_Tag38YV0"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_GEOMETRY")
                 ),
        cms.PSet(record = cms.string("CSCRecoDigiParametersRcd"),
                 tag = cms.string("CSCRECODIGI_Geometry_Tag38YV0"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_GEOMETRY")
                 ),
        cms.PSet(record = cms.string("CSCRecoGeometryRcd"),
                 tag = cms.string("CSCRECO_Geometry_Tag38YV0"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_GEOMETRY")
                 ),
        cms.PSet(record = cms.string("PEcalBarrelRcd"),
                 tag = cms.string("EBRECO_Geometry_Tag38YV0"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_GEOMETRY")
                 ),
        cms.PSet(record = cms.string("GeometryFileRcd"),
                 tag = cms.string("XMLFILE_Geometry_IdealGFlash_Tag38YV0"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_GEOMETRY"),
                 label = cms.untracked.string("IdealGFlash")
                 ),
        cms.PSet(record = cms.string("GeometryFileRcd"),
                 tag = cms.string("XMLFILE_Geometry_Ideal_Tag38YV0"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_GEOMETRY"),
                 label = cms.untracked.string("Ideal")
                 ),
        cms.PSet(record = cms.string("RPCRecoGeometryRcd"),
                 tag = cms.string("RPCRECO_Geometry_Tag38YV0"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_GEOMETRY")
                 ),
        cms.PSet(record = cms.string("DTRecoGeometryRcd"),
                 tag = cms.string("DTRECO_Geometry_Tag38YV0"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_GEOMETRY")
                 ),
        cms.PSet(record = cms.string("PEcalPreshowerRcd"),
                 tag = cms.string("EPRECO_Geometry_Tag38YV0"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_GEOMETRY")
                 ),
        cms.PSet(record = cms.string("GeometryFileRcd"),
                 tag = cms.string("XMLFILE_Geometry_ExtendedGFlash_Tag38YV0"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_GEOMETRY"),
                 label = cms.untracked.string("ExtendedGFlash")
                 ),
        cms.PSet(record = cms.string("GeometryFileRcd"),
                 tag = cms.string("XMLFILE_Geometry_Extended_Tag38YV0"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_GEOMETRY"),
                 label = cms.untracked.string("Extended")
                 ),
        cms.PSet(record = cms.string("IdealGeometryRecord"),
                 tag = cms.string("TKRECO_Geometry_Tag38YV0"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_GEOMETRY")
                 ),
        cms.PSet(record = cms.string("PHcalRcd"),
                 tag = cms.string("HCALRECO_Geometry_Tag38YV0"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_GEOMETRY")
                 )
        )
    return(process)



# End of customisation function definition

process = customise(process)

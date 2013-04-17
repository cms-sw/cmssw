# Auto generated configuration file
# using: 
# Revision: 1.400 
# Source: /local/reps/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: h2mu_Z2nuH_M125GeV_14TeV_cfi -s GEN,SIM --conditions POSTLS161_V12::All --geometry Geometry/GEMGeometry/cmsExtendedGeometryPostLS1plusGEMXML_cfi --datatier GEN-SIM --eventcontent FEVTDEBUG -n 50000 --no_exec --fileout h2mu_Z2nuH_M125GeV_14TeV_GEN_SIM.root
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
    input = cms.untracked.int32(50000)
)

# Input source
process.source = cms.Source("EmptySource")

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.400 $'),
    annotation = cms.untracked.string('h2mu_Z2nuH_M125GeV_14TeV_cfi nevts:50000'),
    name = cms.untracked.string('PyReleaseValidation')
)

# Output definition

process.FEVTDEBUGoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = process.FEVTDEBUGEventContent.outputCommands,
    fileName = cms.untracked.string('h2mu_Z2nuH_M125GeV_14TeV_GEN_SIM.root'),
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

from Configuration.Generator.PythiaUEZ2Settings_cfi import *
process.generator = cms.EDFilter("Pythia6GeneratorFilter",
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    maxEventsToPrint = cms.untracked.int32(0),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    filterEfficiency = cms.untracked.double(1.0),
    crossSection = cms.untracked.double(55000000000.),
    comEnergy = cms.double(14000.0),
    PythiaParameters = cms.PSet(
    pythiaUESettingsBlock,                                 
# set proccess to be simulated
    processParameters = cms.vstring(
            'MSEL=0            ! User defined processes', 
            'MSUB(24)= 1       ! gg->ZH (SM)', 
            'PMAS(25,1)= 125.  ! m_h',
            'PMAS(6,1)= 172.6  ! mass of top quark',
            'PMAS(23,1)=91.187 ! mass of Z',
            'PMAS(24,1)=80.39  ! mass of W',

# Z decay
            'MDME( 174,1) = 0    !Z decay into d dbar', 
            'MDME( 175,1) = 0    !Z decay into u ubar', 
            'MDME( 176,1) = 0    !Z decay into s sbar', 
            'MDME( 177,1) = 0    !Z decay into c cbar', 
            'MDME( 178,1) = 0    !Z decay into b bbar', 
            'MDME( 179,1) = 0    !Z decay into t tbar', 
            'MDME( 182,1) = 0    !Z decay into e- e+', 
            'MDME( 183,1) = 1    !Z decay into nu_e nu_ebar', 
            'MDME( 184,1) = 0    !Z decay into mu- mu+', 
            'MDME( 185,1) = 1    !Z decay into nu_mu nu_mubar', 
            'MDME( 186,1) = 0    !Z decay into tau- tau+', 
            'MDME( 187,1) = 1    !Z decay into nu_tau nu_taubar', 
# Higgs (h0) decay channels
            'MDME(210,1)=0      ! d               dbar',
            'MDME(211,1)=0      ! u               ubar',
            'MDME(212,1)=0      ! s               sbar',
            'MDME(213,1)=0      ! c               cbar',
            'MDME(214,1)=0      ! b               bbar',
            'MDME(215,1)=0      ! t               tbar',
            'MDME(216,1)=0      ! bp              bp bar',
            'MDME(217,1)=0      ! tp              tp bar',
            'MDME(218,1)=0      ! e-              e+',
            'MDME(219,1)=1      ! mu-             mu+',
            'MDME(220,1)=0      ! tau-            tau+',
            'MDME(221,1)=0      ! taup -           taup +',                                    
            'MDME(222,1)=0      ! g               g',      
            'MDME(223,1)=0      ! gamma           gamma',  
            'MDME(224,1)=0      ! gamma           Z0',     
            'MDME(225,1)=0      ! Z0              Z0',     
            'MDME(226,1)=0      ! W+              W-',     
            'MDME(227,1)=0      ! ~chi_10         ~chi_10',
            'MDME(228,1)=0      ! ~chi_20         ~chi_10',
            'MDME(229,1)=0      ! ~chi_20         ~chi_20',
            'MDME(230,1)=0      ! ~chi_30         ~chi_10',
            'MDME(231,1)=0      ! ~chi_30         ~chi_20',
            'MDME(232,1)=0      ! ~chi_30         ~chi_30',
            'MDME(233,1)=0      ! ~chi_40         ~chi_10',
            'MDME(234,1)=0      ! ~chi_40         ~chi_20',
            'MDME(235,1)=0      ! ~chi_40         ~chi_40',        
            'MDME(237,1)=0      ! ~chi_1+         ~chi_1-',
            'MDME(238,1)=0      ! ~chi_1+         ~chi_2-',
            'MDME(239,1)=0      ! ~chi_2+         ~chi_1-',
            'MDME(240,1)=0      ! ~chi_2+         ~chi_2-',
            'MDME(241,1)=0      ! ~d_L            ~d_Lbar',
            'MDME(242,1)=0      ! ~d_R            ~d_Rbar',
            'MDME(243,1)=0      ! ~d_L            ~d_Rbar',
            'MDME(244,1)=0      ! ~d_Lbar         ~d_R',   
            'MDME(245,1)=0      ! ~u_L            ~u_Lbar',
            'MDME(246,1)=0      ! ~u_R            ~u_Rbar',
            'MDME(247,1)=0      ! ~u_L            ~u_Rbar',
            'MDME(248,1)=0      ! ~u_Lbar         ~u_R',   
            'MDME(249,1)=0      ! ~s_L            ~s_Lbar',
            'MDME(250,1)=0      ! ~s_R            ~s_Rbar',
            'MDME(251,1)=0      ! ~s_L            ~s_Rbar',
            'MDME(252,1)=0      ! ~s_Lbar         ~s_R',   
            'MDME(253,1)=0      ! ~c_L            ~c_Lbar',
            'MDME(254,1)=0      ! ~c_R            ~c_Rbar',
            'MDME(255,1)=0      ! ~c_L            ~c_Rbar',
            'MDME(256,1)=0      ! ~c_Lbar         ~c_R',   
            'MDME(257,1)=0      ! ~b_1            ~b_1bar',
            'MDME(258,1)=0      ! ~b_2            ~b_2bar',
            'MDME(259,1)=0      ! ~b_1            ~b_2bar',
            'MDME(260,1)=0      ! ~b_1bar         ~b_2',
            'MDME(261,1)=0      ! ~t_1            ~t_1bar',
            'MDME(262,1)=0      ! ~t_2            ~t_2bar',
            'MDME(263,1)=0      ! ~t_1            ~t_2bar',                                    
            'MDME(264,1)=0      ! ~t_1bar         ~t_2',   
            'MDME(265,1)=0      ! ~e_L-           ~e_L+',  
            'MDME(266,1)=0      ! ~e_R-           ~e_R+',  
            'MDME(267,1)=0      ! ~e_L-           ~e_R+',  
            'MDME(268,1)=0      ! ~e_L+           ~e_R-',  
            'MDME(269,1)=0      ! ~nu_eL          ~nu_eLbar',                                    
            'MDME(270,1)=0      ! ~nu_eR          ~nu_eRbar',                                    
            'MDME(271,1)=0      ! ~nu_eL          ~nu_eRbar',                                    
            'MDME(272,1)=0      ! ~nu_eLbar       ~nu_eR', 
            'MDME(273,1)=0      ! ~mu_L-          ~mu_L+', 
            'MDME(274,1)=0      ! ~mu_R-          ~mu_R+', 
            'MDME(275,1)=0      ! ~mu_L-          ~mu_R+', 
            'MDME(276,1)=0      ! ~mu_L+          ~mu_R-', 
            'MDME(277,1)=0      ! ~nu_muL         ~nu_muLbar',                                    
            'MDME(278,1)=0      ! ~nu_muR         ~nu_muRbar',                                    
            'MDME(279,1)=0      ! ~nu_muL         ~nu_muRbar',                                    
            'MDME(280,1)=0      ! ~nu_muLbar      ~nu_muR',
            'MDME(281,1)=0      ! ~tau_1-         ~tau_1+',
            'MDME(282,1)=0      ! ~tau_2-         ~tau_2+',
            'MDME(283,1)=0      ! ~tau_1-         ~tau_2+',
            'MDME(284,1)=0      ! ~tau_1+         ~tau_2-',
            'MDME(285,1)=0      ! ~nu_tauL        ~nu_tauLbar',
            'MDME(286,1)=0      ! ~nu_tauR        ~nu_tauRbar',                                    
            'MDME(287,1)=0      ! ~nu_tauL        ~nu_tauRbar',                                    
            'MDME(288,1)=0      ! ~nu_tauLbar     ~nu_tauR',                                    
# Higgs boson decays
            'MDME(334,1)=0      ! Higgs(H) decay into d             dbar',
            'MDME(335,1)=0      ! Higgs(H) decay into u             ubar',
            'MDME(336,1)=0      ! Higgs(H) decay into s             sbar',
            'MDME(337,1)=0      ! Higgs(H) decay into c             cbar',
            'MDME(338,1)=0      ! Higgs(H) decay into b             bbar',
            'MDME(339,1)=0      ! Higgs(H) decay into t             tbar',
            'MDME(340,1)=0      ! Higgs(H) decay into b            b bar',
            'MDME(341,1)=0      ! Higgs(H) decay into t            t bar',
            'MDME(342,1)=0      ! Higgs(H) decay into e-              e+',
            'MDME(343,1)=0      ! Higgs(H) decay into mu-            mu+',
            'MDME(344,1)=0      ! Higgs(H) decay into tau-          tau+',
            'MDME(345,1)=0      ! Higgs(H) decay into tau -        tau +',
            'MDME(346,1)=0      ! Higgs(H) decay into g                g',
            'MDME(347,1)=0      ! Higgs(H) decay into gamma        gamma',
            'MDME(348,1)=0      ! Higgs(H) decay into gamma           Z0',
            'MDME(349,1)=0      ! Higgs(H) decay into Z0              Z0',
            'MDME(350,1)=0      ! Higgs(H) decay into W+              W-',
            'MDME(351,1)=0      ! Higgs(H) decay into Z0              h0',
            'MDME(352,1)=0      ! Higgs(H) decay into h0              h0',
            'MDME(353,1)=0      ! Higgs(H) decay into W+              H-',
            'MDME(354,1)=0      ! Higgs(H) decay into H+              W-',
            'MDME(355,1)=0      ! Higgs(H) decay into Z0              A0',
            'MDME(356,1)=0      ! Higgs(H) decay into h0              A0',
            'MDME(357,1)=0      ! Higgs(H) decay into A0              A0',
            'MDME(358,1)=0      ! Higgs(H) decay into ~chi_10         ~chi_10',
            'MDME(359,1)=0      ! Higgs(H) decay into ~chi_20         ~chi_10',
            'MDME(360,1)=0      ! Higgs(H) decay into ~chi_20         ~chi_20',
            'MDME(361,1)=0      ! Higgs(H) decay into ~chi_30         ~chi_10',
            'MDME(362,1)=0      ! Higgs(H) decay into ~chi_30         ~chi_20', 
            'MDME(363,1)=0      ! Higgs(H) decay into ~chi_30         ~chi_30', 
            'MDME(364,1)=0      ! Higgs(H) decay into ~chi_40         ~chi_10', 
            'MDME(365,1)=0      ! Higgs(H) decay into ~chi_40         ~chi_20', 
            'MDME(366,1)=0      ! Higgs(H) decay into ~chi_40         ~chi_30', 
            'MDME(367,1)=0      ! Higgs(H) decay into ~chi_40         ~chi_40', 
            'MDME(368,1)=0      ! Higgs(H) decay into ~chi_1+         ~chi_1-', 
            'MDME(369,1)=0      ! Higgs(H) decay into ~chi_1+         ~chi_2-', 
            'MDME(370,1)=0      ! Higgs(H) decay into ~chi_2+         ~chi_1-', 
            'MDME(371,1)=0      ! Higgs(H) decay into ~chi_2+         ~chi_2-', 
            'MDME(372,1)=0      ! Higgs(H) decay into ~d_L            ~d_Lbar', 
            'MDME(373,1)=0      ! Higgs(H) decay into ~d_R            ~d_Rbar', 
            'MDME(374,1)=0      ! Higgs(H) decay into ~d_L            ~d_Rbar', 
            'MDME(375,1)=0      ! Higgs(H) decay into ~d_Lbar         ~d_R', 
            'MDME(376,1)=0      ! Higgs(H) decay into ~u_L            ~u_Lbar', 
            'MDME(377,1)=0      ! Higgs(H) decay into ~u_R            ~u_Rbar', 
            'MDME(378,1)=0      ! Higgs(H) decay into ~u_L            ~u_Rbar', 
            'MDME(379,1)=0      ! Higgs(H) decay into ~u_Lbar         ~u_R', 
            'MDME(380,1)=0      ! Higgs(H) decay into ~s_L            ~s_Lbar', 
            'MDME(381,1)=0      ! Higgs(H) decay into ~s_R            ~s_Rbar', 
            'MDME(382,1)=0      ! Higgs(H) decay into ~s_L            ~s_Rbar', 
            'MDME(383,1)=0      ! Higgs(H) decay into ~s_Lbar         ~s_R', 
            'MDME(384,1)=0      ! Higgs(H) decay into ~c_L            ~c_Lbar', 
            'MDME(385,1)=0      ! Higgs(H) decay into ~c_R            ~c_Rbar', 
            'MDME(386,1)=0      ! Higgs(H) decay into ~c_L            ~c_Rbar', 
            'MDME(387,1)=0      ! Higgs(H) decay into ~c_Lbar         ~c_R', 
            'MDME(388,1)=0      ! Higgs(H) decay into ~b_1            ~b_1bar', 
            'MDME(389,1)=0      ! Higgs(H) decay into ~b_2            ~b_2bar', 
            'MDME(390,1)=0      ! Higgs(H) decay into ~b_1            ~b_2bar', 
            'MDME(391,1)=0      ! Higgs(H) decay into ~b_1bar         ~b_2', 
            'MDME(392,1)=0      ! Higgs(H) decay into ~t_1            ~t_1bar', 
            'MDME(393,1)=0      ! Higgs(H) decay into ~t_2            ~t_2bar', 
            'MDME(394,1)=0      ! Higgs(H) decay into ~t_1            ~t_2bar', 
            'MDME(395,1)=0      ! Higgs(H) decay into ~t_1bar         ~t_2', 
            'MDME(396,1)=0      ! Higgs(H) decay into ~e_L-           ~e_L+', 
            'MDME(397,1)=0      ! Higgs(H) decay into ~e_R-           ~e_R+', 
            'MDME(398,1)=0      ! Higgs(H) decay into ~e_L-           ~e_R+', 
            'MDME(399,1)=0      ! Higgs(H) decay into ~e_L+           ~e_R-', 
            'MDME(400,1)=0      ! Higgs(H) decay into ~nu_eL          ~nu_eLbar', 
            'MDME(401,1)=0      ! Higgs(H) decay into ~nu_eR          ~nu_eRbar', 
            'MDME(402,1)=0      ! Higgs(H) decay into ~nu_eL          ~nu_eRbar', 
            'MDME(403,1)=0      ! Higgs(H) decay into ~nu_eLbar       ~nu_eR', 
            'MDME(404,1)=0      ! Higgs(H) decay into ~mu_L-          ~mu_L+', 
            'MDME(405,1)=0      ! Higgs(H) decay into ~mu_R-          ~mu_R+', 
            'MDME(406,1)=0      ! Higgs(H) decay into ~mu_L-          ~mu_R+', 
            'MDME(407,1)=0      ! Higgs(H) decay into ~mu_L+          ~mu_R-',
            'MDME(408,1)=0      ! Higgs(H) decay into ~nu_muL         ~nu_muLbar',
            'MDME(409,1)=0      ! Higgs(H) decay into ~nu_muR         ~nu_muRbar', 
            'MDME(410,1)=0      ! Higgs(H) decay into ~nu_muL         ~nu_muRbar',
            'MDME(411,1)=0      ! Higgs(H) decay into ~nu_muLbar      ~nu_muR',
            'MDME(412,1)=0      ! Higgs(H) decay into ~tau_1-         ~tau_1+',
            'MDME(413,1)=0      ! Higgs(H) decay into ~tau_2-         ~tau_2+',
            'MDME(414,1)=0      ! Higgs(H) decay into ~tau_1-         ~tau_2+',
            'MDME(415,1)=0      ! Higgs(H) decay into ~tau_1+         ~tau_2-',
            'MDME(416,1)=0      ! Higgs(H) decay into ~nu_tauL        ~nu_tauLbar', 
            'MDME(417,1)=0      ! Higgs(H) decay into ~nu_tauR        ~nu_tauRbar',
            'MDME(418,1)=0      ! Higgs(H) decay into ~nu_tauL        ~nu_tauRbar',
            'MDME(419,1)=0      ! Higgs(H) decay into ~nu_tauLbar     ~nu_tauR'
        ),                            
        # This is a vector of ParameterSet names to be read, in this order
        parameterSets = cms.vstring('pythiaUESettings', 
            'processParameters')
    )
)

process.ProductionFilterSequence = cms.Sequence(process.generator)

# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGoutput_step = cms.EndPath(process.FEVTDEBUGoutput)

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.simulation_step,process.endjob_step,process.FEVTDEBUGoutput_step)
# filter all path with the production filter sequence
for path in process.paths:
	getattr(process,path)._seq = process.ProductionFilterSequence * getattr(process,path)._seq

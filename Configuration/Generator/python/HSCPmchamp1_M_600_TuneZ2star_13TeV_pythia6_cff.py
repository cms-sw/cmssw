FLAVOR = 'stau'
COM_ENERGY = 13000. 
MASS_POINT = 600   # GeV
CHARGE = 1   # electron charge/3
PROCESS_FILE = 'SimG4Core/CustomPhysics/data/RhadronProcessList.txt'
PARTICLE_FILE = 'Configuration/Generator/data/particles_HIP%d_%s_%d_GeV.txt' % (CHARGE, FLAVOR, MASS_POINT)
SLHA_FILE = 'None'
PDT_FILE = 'Configuration/Generator/data/hscppythiapdtHIP%d%s%d.tbl'  % (CHARGE, FLAVOR, MASS_POINT)
USE_REGGE = False

hipMass = float (MASS_POINT)

import FWCore.ParameterSet.Config as cms

from Configuration.Generator.PythiaUEZ2starSettings_cfi import *

generator = cms.EDFilter("Pythia6GeneratorFilter",
    filterEfficiency = cms.untracked.double(1.),
    comEnergy = cms.double(COM_ENERGY),
    crossSection = cms.untracked.double(-1),
    maxEventsToPrint = cms.untracked.int32(0),
                         
	PythiaParameters = cms.PSet(
	    pythiaUESettingsBlock,
	 	processParameters = cms.vstring(
 	      'MSEL=0          ! User defined processes',
		  'MSUB(1)=1 !',
		  'MSTP(43)    = 3   ! complete Z0/gamma* interference',
		  'MSTP(1)=4 !fourth generation',
		  'CKIN(1)=%f !min sqrt(s hat)' % hipMass,
		  'CKIN(2)= -1  ! (no) max sqrt(s hat) (GeV)', 
                  'KCHG(17,1)=%i !charge of tauprime' % CHARGE,
		  'PMAS(17,1)=%f !tauprime mass' % hipMass,
		  'MDME(174,1) = 0   !Z decay into d dbar', 
		  'MDME(175,1) = 0   !Z decay into u ubar', 
		  'MDME(176,1) = 0   !Z decay into s sbar', 
		  'MDME(177,1) = 0   !Z decay into c cbar', 
		  'MDME(178,1) = 0   !Z decay into b bbar', 
		  'MDME(179,1) = 0   !Z decay into t tbar', 
		  'MDME(180,1) = 0   !Z decay into bprime bprimebar', 
		  'MDME(181,1) = 0   !Z decay into tprime tprimebar', 
		  'MDME(182,1) = 0   !Z decay into e- e+', 
		  'MDME(183,1) = 0   !Z decay into nu_e nu_ebar', 
		  'MDME(184,1) = 0   !Z decay into mu- mu+', 
		  'MDME(185,1) = 0   !Z decay into nu_mu nu_mubar', 
		  'MDME(186,1) = 0   !Z decay into tau- tau+', 
		  'MDME(187,1) = 0   !Z decay into nu_tau nu_taubar',
		  'MDME(188,1) = 1   !Z decay into tauprime tauprimebar',
		  'MDME(189,1) = 0   !Z decay into nu_tauprime nu_tauprimebar',
		  'MDCY(17,1)=0    ! set tauprime stable',
		  'MWID(17)=0      ! set tauprime width 0'
		  ),
    parameterSets = cms.vstring(

    'pythiaUESettings', 'processParameters'),
    
    )
 )
                         
generator.hscpFlavor = cms.untracked.string(FLAVOR)
generator.massPoint = cms.untracked.int32(MASS_POINT)
generator.slhaFile = cms.untracked.string(SLHA_FILE)
generator.processFile = cms.untracked.string(PROCESS_FILE)
generator.particleFile = cms.untracked.string(PARTICLE_FILE)
generator.pdtFile = cms.FileInPath(PDT_FILE)
generator.useregge = cms.bool(USE_REGGE)

ProductionFilterSequence = cms.Sequence(generator)

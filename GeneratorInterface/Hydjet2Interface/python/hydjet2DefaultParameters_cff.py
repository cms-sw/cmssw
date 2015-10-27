import FWCore.ParameterSet.Config as cms

from Configuration.Generator.PythiaUESettings_cfi import *

collisionParameters2760GeV = cms.PSet(
	fAw = cms.double(208.0), 	# beam/target atomic number
	fSqrtS = cms.double(2760.0), 	#
	fUmax 	= cms.double(1.265), 	# Maximal transverse flow rapidity at thermal freeze-out for central collisions, fUmax
	fPtmin 	= cms.double(8.2), 	# Minimal pt of parton-parton scattering in PYTHIA event, fPtmin [GeV/c] 
	fT0 = cms.double(1.), 		# Initial QGP temperature for central Pb+Pb collisions in mid-rapidity, fT0 [GeV]; allowed range [0.2,2.0]GeV;

	### Volume parameters at thermal freeze-out ###
	fTau 	= cms.double(12.2), 	# Proper time proper at thermal freeze-out for central collisions, fTau [fm/c]
	fR 	= cms.double(13.45), 	# Maximal transverse radius at thermal freeze-out for central collisions, fR [fm]
)

collisionParameters5020GeV = cms.PSet(
	fAw = cms.double(208.0), ## beam/target atomic number
	fSqrtS = cms.double(5020.0),
	fUmax 	= cms.double(1.35), 	# Maximal transverse flow rapidity at thermal freeze-out for central collisions, fUmax
	fPtmin 	= cms.double(10.), 	# Minimal pt of parton-parton scattering in PYTHIA event, fPtmin [GeV/c] 
	fT0 = cms.double(1.1), 		# Initial QGP temperature for central Pb+Pb collisions in mid-rapidity, fT0 [GeV]; allowed range [0.2,2.0]GeV;

	### Volume parameters at thermal freeze-out ###
	fTau 	= cms.double(13.2), 	# Proper time proper at thermal freeze-out for central collisions, fTau [fm/c]
	fR 	= cms.double(13.9), 	# Maximal transverse radius at thermal freeze-out for central collisions, fR [fm]
)

qgpParameters = cms.PSet(
	fTau0 	= cms.double(0.1), 	# Proper QGP formation time in fm/c, fTau0 (0.01<fTau0<10)
	fNf 	= cms.int32(0), 	# Number of active quark flavours in QGP, fNf (0, 1, 2 or 3)
)

hydjet2Parameters = cms.PSet(
	### Thermodinamic parameters at chemical freez-out ###
	fTMuType 	= cms.double(0.), 	# Flag to use calculated T_ch, mu_B and mu_S as a function of fSqrtS, fTMuType (=0 user's ones, >0 calculated) 
	fT 	= cms.double(0.165), 	# Temperature at chemical freeze-out, fT [GeV]
	fMuB 	= cms.double(0.), 	# Chemical baryon potential per unit charge, fMuB [GeV]
	fMuS 	= cms.double(0.), 	# Chemical strangeness potential per unit charge, fMuS [GeV]  
	fMuC 	= cms.double(0.), 	# Chemical charm potential per unit charge, fMuC [GeV] (used if charm production is turned on)  
	fMuI3 	= cms.double(0.), 	# Chemical isospin potential per unit charge, fMuI3 [GeV]   

   	### Thermodinamic parameters at thermal freez-out ###
	fThFO 	= cms.double(0.105), 	# Temperature at thermal freeze-out, fTthFO [GeV]
	fMu_th_pip 	= cms.double(0.), 	# Chemical potential of pi+ at thermal freeze-out, fMu_th_pip [GeV] 

	### Volume parameters at thermal freeze-out ###
	fSigmaTau 	= cms.double(3.5), 	# Duration of emission at thermal freeze-out for central collisions, fSigmaTau [fm/c]

	### Strangeness suppression factor ###
	fCorrS 	= cms.double(1.), 	# Strangeness supression factor gamma_s with fCorrS value (0<fCorrS <=1, if fCorrS <= 0 then it is calculated)

	### Maximal longitudinal flow rapidity at thermal freeze-out ###
	fYlmax 	= cms.double(4.5), 	# Maximal longitudinal flow rapidity at thermal freeze-out, fYlmax
	

	### Anizotropy parameter at thermal freeze-out ###
	fIfDeltaEpsilon = cms.double(1.), 	# Flag to specify fDelta and fEpsilon values, fIfDeltaEpsilon (=0 user's ones, >=1 calculated) 
	fDelta 	= cms.double(0.1), 	# Momentum azimuthal anizotropy parameter at thermal freeze-out, fDelta
	fEpsilon 	= cms.double(0.05), 	# Spatial azimuthal anisotropy parameter at thermal freeze-out, fEpsilon

  	### Decays ###
	fDecay 	= cms.int32(1), 	# Flag to switch on/off hadron decays, fDecay (=0 decays off, >=1 decays on)
	fWeakDecay 	= cms.double(0.000000000000001), 	# Low decay width threshold fWeakDecay[GeV]: width<fWeakDecay decay off, width>=fDecayWidth decay on; can be used to switch off weak decays
	
  	### Charm ###
	fCharmProd 	= cms.int32(1), 	# Flag to include thermal charm production, fIcharm (=0 no charm production, >=1 charm production) 
	fCorrC 	= cms.double(-1.), 	# Charmness enhancement factor gamma_c with fCorrC value (fCorrC >0, if fCorrC<0 then it is calculated)



	fEtaType 	= cms.double(1.), 	# Flag to choose longitudinal flow rapidity distribution, fEtaType (=0 uniform, >0 Gaussian with the dispersion Ylmax)
	fIshad 	= cms.int32(1), 	# Flag to switch on/off nuclear shadowing, fIshad (0 shadowing off, 1 shadowing on)	
	fPyhist 	= cms.int32(0), 	# Flag to suppress the output of particle history from PYTHIA, fPyhist (=1 only final state particles; =0 full particle history from PYTHIA)
	fIenglu 	= cms.int32(0), 	# Flag to fix type of partonic energy loss, fIenglu (0 radiative and collisional loss, 1 radiative loss only, 2 collisional loss only)
	fIanglu 	= cms.int32(1), 	# Flag to fix type of angular distribution of in-medium emitted gluons, fIanglu (0 small-angular, 1 wide-angular, 2 collinear).
	embeddingMode = cms.bool(False),
	rotateEventPlane = cms.bool(True)
	 
)

PythiaDefaultBlock = cms.PSet(
	pythiaUESettingsBlock,
	hydjet2PythiaDefault = cms.vstring(
		'MSEL=1',		# ! type of hard QCD production process
	   	'MSTU(21) = 1',	# ! controle parameter to avoid stopping run
		'PARU(14)=1.', 	# ! tolerance parameter to adjust fragmentation'
		#'MSTP(81)=1',!in Q2O  	# ! pp multiple scattering on
		'MSTJ(21) = 1',	# ! hadron decays on (if off - decays by FASTMC decayer) 
		'MSTP(2) = 1',	# ! which order running alphaS 
		#'MSTP(33) = 0',!in Q2O	# ! incluion of k factor in cross section
		#'mstp(51)=7',!in Q2O	# ! PDF set: structure function chosen - CTEQ5M pdf
		#'MSTP(82) = 4',!in Q2O	# ! defines the multi-parton model
		#'PARP(67) = 2.65',!Q2O	# ! amount of initial-state radiation
		#'PARP(82) = 1.9',!Q2O	# ! pt cutoff for multiparton int32eractions 
		#'MSTJ(11) = 5',!Q2O	# ! Choice of the fragmentation function 
		'MSTJ(22)=2',	# ! particle decays if lifetime < parj(71)
		'PARJ(71)=10.',	# ! ctau=10 mm 
		'MSTP(52) = 1',	# ! NO LAPDF		
		'mstp(122)=0'	# ! no printout of Pythia initialization information hereinafter 
	),
	ProQ2Otune = cms.vstring(  
		'mstp(51)=7',	# ! PDF set: structure function chosen - CTEQ5M pdf                                      
      		'mstp(3)=2',	# ! QCD switch for choice of LambdaQCD           
      		'parp(62)=2.9',	# ! ISR IR cutoff                                
      		'parp(64)=0.14',	# ! ISR renormalization scale prefactor          
      		'parp(67)=2.65',	# ! ISR Q2max factor                             
      		'mstp(68)=3',	# ! ISR phase space choice & ME corrections      
      		'parp(71)=4.',	# ! FSR Q2max factor for non-s-channel procs     
      		'parj(81)=0.29',	# ! FSR Lambda_QCD scale                         
      		'parj(82)=1.65',	# ! FSR IR cutoff                                
      		'mstp(33)=0',	# ! "K" switch for K-factor on/off & type        
      		'mstp(81)=1',	# ! UE model                                     
      		'parp(82)=1.9',	# ! UE IR cutoff at reference ecm                
      		'parp(89)=1800.',	# ! UE IR cutoff reference ecm                   
      		'parp(90)=0.22',	# ! UE IR cutoff ecm scaling power               
      		'mstp(82)=4',	# ! UE hadron transverse mass distribution       
      		'parp(83)=0.83',	# ! UE mass distribution parameter               
      		'parp(84)=0.6',	# ! UE mass distribution parameter              
      		'parp(85)=0.86',	# ! UE gg colour correlated fraction             
      		'parp(86)=0.93',	# ! UE total gg fraction                         
      		'mstp(91)=1',	# ! BR primordial kT distribution                
      		'parp(91)=2.1',	# ! BR primordial kT width <|kT|>                
      		'parp(93)=5.',	# ! BR primordial kT UV cutoff               
      		'mstj(11)=5',	# ! HAD choice of fragmentation function(s)      
      		'parj(1)=0.073',	# ! HAD diquark suppression                      
      		'parj(2)=0.2',	# ! HAD strangeness suppression                  
      		'parj(3)=0.94',	# ! HAD strange diquark suppression              
      		'parj(4)=0.032',	# ! HAD vector diquark suppression               
      		'parj(11)=0.31',	# ! HAD P(vector meson), u and d only            
      		'parj(12)=0.4',	# ! HAD P(vector meson), contains s              
      		'parj(13)=0.54',	# ! HAD P(vector meson), heavy quarks            
      		'parj(21)=0.325',	# ! HAD fragmentation pT                         
      		'parj(25)=0.63',	# ! HAD eta0 suppression                        
      		'parj(26)=0.12',	# ! HAD eta0' suppression                       
      		'parj(41)=0.5',	# ! HAD string parameter a                       
      		'parj(42)=0.6',	# ! HAD string parameter b                       
      		'parj(46)=1.',	# ! HAD Lund(=0)-Bowler(=1) rQ (rc)              
      		'parj(47)=0.67'	# ! HAD Lund(=0)-Bowler(=1) rb            

	),
	ppJets = cms.vstring('MSEL=1'),# ! QCD hight pT processes
	customProcesses = cms.vstring('MSEL=0'),# ! User processes
	pythiaJets = cms.vstring(
		'MSUB(11)=1', # q+q->q+q
		'MSUB(12)=1', # q+qbar->q+qbar
		'MSUB(13)=1', # q+qbar->g+g
		'MSUB(28)=1', # q+g->q+g
		'MSUB(53)=1', # g+g->q+qbar
		'MSUB(68)=1' # g+g->g+g
	),
	pythiaPromptPhotons = cms.vstring(
		'MSUB(14)=1', # q+qbar->g+gamma
		'MSUB(18)=1', # q+qbar->gamma+gamma
		'MSUB(29)=1', # q+g->q+gamma
		'MSUB(114)=1', # g+g->gamma+gamma
		'MSUB(115)=1' # g+g->g+gamma
	),
	pythiaWeakBosons = cms.vstring(
		'MSUB(1)=1',
		'MSUB(2)=1'
	),
	pythiaZjets = cms.vstring(
		'MSUB(15)=1',
		'MSUB(30)=1'
	),
	pythiaCharmoniumNRQCD = cms.vstring(
		'MSUB(421) = 1',
		'MSUB(422) = 1',
		'MSUB(423) = 1',
		'MSUB(424) = 1',
		'MSUB(425) = 1',
		'MSUB(426) = 1',
		'MSUB(427) = 1',
		'MSUB(428) = 1',
		'MSUB(429) = 1',
		'MSUB(430) = 1',
		'MSUB(431) = 1',
		'MSUB(432) = 1',
		'MSUB(433) = 1',
		'MSUB(434) = 1',
		'MSUB(435) = 1',
		'MSUB(436) = 1',
		'MSUB(437) = 1',
		'MSUB(438) = 1',
		'MSUB(439) = 1'
	),
	pythiaBottomoniumNRQCD = cms.vstring(
		'MSUB(461) = 1',
		'MSUB(462) = 1',
		'MSUB(463) = 1',
		'MSUB(464) = 1',
		'MSUB(465) = 1',
		'MSUB(466) = 1',
		'MSUB(467) = 1',
		'MSUB(468) = 1',
		'MSUB(469) = 1',
		'MSUB(470) = 1',
		'MSUB(471) = 1',
		'MSUB(472) = 1',
		'MSUB(473) = 1',
		'MSUB(474) = 1',
		'MSUB(475) = 1',
		'MSUB(476) = 1',
		'MSUB(477) = 1',
		'MSUB(478) = 1',
		'MSUB(479) = 1',
	),
	pythiaQuarkoniaSettings = cms.vstring(
		'PARP(141)=1.16', # Matrix Elements
		'PARP(142)=0.0119',
		'PARP(143)=0.01',
		'PARP(144)=0.01',
		'PARP(145)=0.05',
		'PARP(146)=9.28',
		'PARP(147)=0.15',
		'PARP(148)=0.02',
		'PARP(149)=0.02',
		'PARP(150)=0.085',
		# Meson spin
		'PARJ(13)=0.60',
		'PARJ(14)=0.162',
		'PARJ(15)=0.018',
		'PARJ(16)=0.054',
		# Polarization
		'MSTP(145)=0',
		'MSTP(146)=0',
		'MSTP(147)=0',
		'MSTP(148)=1',
		'MSTP(149)=1',
		# Chi_c branching ratios
		'BRAT(861)=0.202',
		'BRAT(862)=0.798',
		'BRAT(1501)=0.013',
		'BRAT(1502)=0.987',
		'BRAT(1555)=0.356',
		'BRAT(1556)=0.644'
	),
	pythiaZtoMuons = cms.vstring(
		"MDME(174,1)=0", # !Z decay into d dbar,
		"MDME(175,1)=0", # !Z decay into u ubar,
		"MDME(176,1)=0", # !Z decay into s sbar,
		"MDME(177,1)=0", # !Z decay into c cbar,
		"MDME(178,1)=0", # !Z decay into b bbar,
		"MDME(179,1)=0", # !Z decay into t tbar,
		"MDME(182,1)=0", # !Z decay into e- e+,
		"MDME(183,1)=0", # !Z decay into nu_e nu_ebar,
		"MDME(184,1)=1", # !Z decay into mu- mu+,
		"MDME(185,1)=0", # !Z decay into nu_mu nu_mubar,
		"MDME(186,1)=0", # !Z decay into tau- tau+,
		"MDME(187,1)=0" # !Z decay into nu_tau nu_taubar
	),
	pythiaZtoElectrons = cms.vstring(
		"MDME(174,1)=0", # !Z decay into d dbar,
		"MDME(175,1)=0", # !Z decay into u ubar,
		"MDME(176,1)=0", # !Z decay into s sbar,
		"MDME(177,1)=0", # !Z decay into c cbar,
		"MDME(178,1)=0", # !Z decay into b bbar,
		"MDME(179,1)=0", # !Z decay into t tbar,
		"MDME(182,1)=1", # !Z decay into e- e+,
		"MDME(183,1)=0", # !Z decay into nu_e nu_ebar,
		"MDME(184,1)=0", # !Z decay into mu- mu+,
		"MDME(185,1)=0", # !Z decay into nu_mu nu_mubar,
		"MDME(186,1)=0", # !Z decay into tau- tau+,
		"MDME(187,1)=0" # !Z decay into nu_tau nu_taubar
	),
	pythiaZtoMuonsAndElectrons = cms.vstring(
		"MDME(174,1)=0", # !Z decay into d dbar,
		"MDME(175,1)=0", # !Z decay into u ubar,
		"MDME(176,1)=0", # !Z decay into s sbar,
		"MDME(177,1)=0", # !Z decay into c cbar,
		"MDME(178,1)=0", # !Z decay into b bbar,
		"MDME(179,1)=0", # !Z decay into t tbar,
		"MDME(182,1)=1", # !Z decay into e- e+,
		"MDME(183,1)=0", # !Z decay into nu_e nu_ebar,
		"MDME(184,1)=1", # !Z decay into mu- mu+,
		"MDME(185,1)=0", # !Z decay into nu_mu nu_mubar,
		"MDME(186,1)=0", # !Z decay into tau- tau+,
		"MDME(187,1)=0" # !Z decay into nu_tau nu_taubar
	),
	pythiaUpsilonToMuons = cms.vstring(
		'BRAT(1034) = 0 ', # switch off',
		'BRAT(1035) = 1 ', # switch on',
		'BRAT(1036) = 0 ', # switch off',
		'BRAT(1037) = 0 ', # switch off',
		'BRAT(1038) = 0 ', # switch off',
		'BRAT(1039) = 0 ', # switch off',
		'BRAT(1040) = 0 ', # switch off',
		'BRAT(1041) = 0 ', # switch off',
		'BRAT(1042) = 0 ', # switch off',
		'MDME(1034,1) = 0 ', # switch off',
		'MDME(1035,1) = 1 ', # switch on',
		'MDME(1036,1) = 0 ', # switch off',
		'MDME(1037,1) = 0 ', # switch off',
		'MDME(1038,1) = 0 ', # switch off',
		'MDME(1039,1) = 0 ', # switch off',
		'MDME(1040,1) = 0 ', # switch off',
		'MDME(1041,1) = 0 ', # switch off',
		'MDME(1042,1) = 0 ', # switch off'
	),
	pythiaJpsiToMuons = cms.vstring(
		'BRAT(858) = 0 ', # switch off',
		'BRAT(859) = 1 ', # switch on',
		'BRAT(860) = 0 ', # switch off',
		'MDME(858,1) = 0 ', # switch off',
		'MDME(859,1) = 1 ', # switch on',
		'MDME(860,1) = 0 ', # switch off'
	),
	pythiaMuonCandidates = cms.vstring(
		'CKIN(3)=20',
		'MSTJ(22)=2',
		'PARJ(71)=40.'
	),
	myParameters = cms.vstring('MDCY(310,1)=0')
)

import FWCore.ParameterSet.Config as cms

from Configuration.Generator.PythiaUESettings_cfi import *
generator = cms.EDFilter("Pythia6GeneratorFilter",
    pythiaPylistVerbosity = cms.untracked.int32(0),
    filterEfficiency = cms.untracked.double(0.141),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    crossSection = cms.untracked.double(9090000.0),
    comEnergy = cms.double(10000.0),
    maxEventsToPrint = cms.untracked.int32(0),
    PythiaParameters = cms.PSet(
        pythiaUESettingsBlock,
        processParameters = cms.vstring(
		'MSEL=62          ! Quarkonia NRQCD bb',
		'CKIN(3)=20.       ! Min pthard',
		'CKIN(4)=-1.      ! Max pthard', 
		'MDME(1034,1)=0   ! 0.025200    e- e+',
		'MDME(1035,1)=1   ! 0.024800    mu- mu+',                   
		'MDME(1036,1)=0   ! 0.026700    tau- tau+',
		'MDME(1037,1)=0   ! 0.015000    d dbar',
		'MDME(1038,1)=0   ! 0.045000    u ubar',
		'MDME(1039,1)=0   ! 0.015000    s sbar',
		'MDME(1040,1)=0   ! 0.045000    c cbar',
		'MDME(1041,1)=0   ! 0.774300    g g g',
		'MDME(1042,1)=0   ! 0.029000    gamma g',
		'MSTP(142)=2      ! turns on the PYEVWT Pt re-weighting routine',
		'PARJ(13)=0.750   ! probability that a c or b meson has S=1',
		'PARJ(14)=0.162   ! probability that a meson with S=0 is produced with L=1, J=1',
		'PARJ(15)=0.018   ! probability that a meson with S=1 is produced with L=1, J=0',
		'PARJ(16)=0.054   ! probability that a meson with S=1 is produced with L=1, J=1',
		'MSTP(145)=0      !  choice of polarization',
		'MSTP(146)=0      ! choice of polarization frame ONLY when mstp(145)=1',
		'MSTP(147)=0      ! particular helicity or density matrix component when mstp(145)=1',
		'MSTP(148)=1      ! possibility to allow for final-state shower evolution, extreme case!',
		'MSTP(149)=1      ! if mstp(148)=1, it determines the kinematics of the QQ~3S1(8)->QQ~3S1(8)+g branching',
		'PARP(141)=1.16   ! New values for COM matrix elements',
		'PARP(142)=0.0119 ! New values for COM matrix elements', 
		'PARP(143)=0.01   ! New values for COM matrix elements', 
		'PARP(144)=0.01   ! New values for COM matrix elements', 
		'PARP(145)=0.05   ! New values for COM matrix elements',  
		'PARP(146)=9.28   ! New values for COM matrix elements', 
		'PARP(147)=0.15   ! New values for COM matrix elements', 
		'PARP(148)=0.02   ! New values for COM matrix elements', 
		'PARP(149)=0.02   ! New values for COM matrix elements', 
		'PARP(150)=0.09   ! New values for COM matrix elements'
	    ),
        # This is a vector of ParameterSet names to be read, in this order
        parameterSets = cms.vstring('pythiaUESettings', 
            'processParameters', 
            'CSAParameters'),
        CSAParameters = cms.vstring('CSAMODE = 6     ! cross-section reweighted quarkonia')
    )
)

oniafilter = cms.EDFilter("PythiaFilter",
    Status = cms.untracked.int32(2),
    MaxEta = cms.untracked.double(1000.0),
    MinEta = cms.untracked.double(-1000.0),
    MinPt = cms.untracked.double(0.0),
    ParticleID = cms.untracked.int32(553)
)

mumugenfilter = cms.EDFilter("MCParticlePairFilter",
    Status = cms.untracked.vint32(1, 1),
    MinPt = cms.untracked.vdouble(2.5, 2.5),
    MaxEta = cms.untracked.vdouble(2.5, 2.5),
    MinEta = cms.untracked.vdouble(-2.5, -2.5),
    ParticleCharge = cms.untracked.int32(-1),
    ParticleID1 = cms.untracked.vint32(13),
    ParticleID2 = cms.untracked.vint32(13)
)

ProductionFilterSequence = cms.Sequence(generator*oniafilter*mumugenfilter)

import FWCore.ParameterSet.Config as cms

#
# this is a leftover from an "intermediate" version of cmsDriver
# we keep it here so far, as an example for 311/312 and 32x, 
# but starting 313 the machinery has been updated, and also 330pre4
#
#source = cms.Source("LHESource",
#    fileNames = cms.untracked.vstring(
#    'file:../../../GeneratorInterface/Pythia6Interface/test/ttbar_5flavours_xqcut20_10TeV.lhe')
#)

from Configuration.Generator.PythiaUESettings_cfi import *
generator = cms.EDFilter("Pythia6HadronizerFilter",
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    maxEventsToPrint = cms.untracked.int32(0),
    pythiaPylistVerbosity = cms.untracked.int32(0),
    filterEfficiency = cms.untracked.double(0.254),
    comEnergy = cms.double(10000.0),
    PythiaParameters = cms.PSet(
        pythiaUESettingsBlock,
        processParameters = cms.vstring('MSEL=0         ! User defined processes',
                        'PMAS(5,1)=4.4   ! b quark mass',
                        'PMAS(6,1)=172.4 ! t quark mass',
                        'MSTJ(1)=1       ! Fragmentation/hadronization on or off',
                        'MSTP(61)=1      ! Parton showering on or off'),
        # This is a vector of ParameterSet names to be read, in this order
        parameterSets = cms.vstring('pythiaUESettings', 
            'processParameters')
    ),
    jetMatching = cms.untracked.PSet(
       scheme = cms.string("Madgraph"),
       mode = cms.string("auto"),       # soup, or "inclusive" / "exclusive"
       MEMAIN_etaclmax = cms.double(5.0),
       MEMAIN_qcut = cms.double(30.0),
       MEMAIN_minjets = cms.int32(-1),
       MEMAIN_maxjets = cms.int32(-1),
    )
)

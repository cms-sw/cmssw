#CROSS_SECTION = 1 # pb
#MASS_HIGGS = 125 # in GeV 
#MASS_HIGGS = 125 # in GeV 
#WIDTH_HIGGS   = 0.027*MASS_HIGGS # Same as default for id=35
#MASS_X = 12 # in GeV 
#MASS_X = 12 # in GeV 
#CTAU_X = 900 # in mm  
#WIDTH_X = 2.1925219866666667e-16 # in mm  
 
import FWCore.ParameterSet.Config as cms
 
from Configuration.Generator.Pythia8CommonSettings_cfi import *
from Configuration.Generator.MCTunes2017.PythiaCP5Settings_cfi import *
from Configuration.Generator.PSweightsPythia.PythiaPSweightsSettings_cfi import *
 
generator = cms.EDFilter("Pythia8ConcurrentGeneratorFilter",
    pythiaPylistVerbosity = cms.untracked.int32(1),
    filterEfficiency = cms.untracked.double(1),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    comEnergy = cms.double(13600.),
    crossSection = cms.untracked.double(CROSS_SECTION),
    maxEventsToPrint = cms.untracked.int32(10),
    PythiaParameters = cms.PSet(
        pythia8CommonSettingsBlock,
#        pythia8CUEP8M1SettingsBlock, # Old PYTHIA tune
        pythia8CP5SettingsBlock,
        pythia8PSweightsSettingsBlock,
        processParameters = cms.vstring(
            "Higgs:useBSM = on",
            "HiggsBSM:all = off",
# Gluon-fusion production only
#            "HiggsBSM:ffbar2H2 = on",
            "HiggsBSM:gg2H2 = on",
# Long-lived scalar decaying to nunu
            "6000111:new = LL_nunu LLbar_nunu 1 0 0",
            "6000111:m0 = %s" % MASS_X,
            "6000111:mWidth = %s" % WIDTH_X,
            "6000111:tau0 = %s" % CTAU_X,
            "6000111:isResonance = on",
            "6000111:mayDecay = on",
            "6000111:oneChannel = 1  0.333 100 12 -12",
            "6000111:addChannel = 1  0.333 100 14 -14",
            "6000111:addChannel = 1  0.333 100 16 -16",
# Long-lived scalar decaying only to bb
            "6000113:new = LL_b LLbar_b 1 0 0",
            "6000113:m0 = %s" % MASS_X,
            "6000113:mWidth = %s" % WIDTH_X,
            "6000113:tau0 = %s" % CTAU_X,
            "6000113:isResonance = on",
            "6000113:mayDecay = on",
            "6000113:oneChannel = 1 1.0 100 5 -5",
# Shut down H0 decays to ordinary particles and A0
            "35:m0 = %s" % MASS_HIGGS,
            "35:mWidth = %s" % WIDTH_HIGGS,
            "35:2:bRatio = 0.0",
            "35:3:bRatio = 0.0",
            "35:4:bRatio = 0.0",
            "35:5:bRatio = 0.0",
            "35:7:bRatio = 0.0",
            "35:8:bRatio = 0.0",
# Keep small coupling of H0 to gluons in order to be produced            
#            "35:9:bRatio = 0.0",
            "35:10:bRatio= 0.0",
            "35:12:bRatio= 0.0",
            "35:13:bRatio= 0.0",
            "35:15:bRatio= 0.0", # h0(25)h0(25); would be open at high m(H0)
            "35:18:bRatio= 0.0", # Z0(23)A0(36); would be open at high m(H0)
            "35:36:bRatio= 0.0", # A0(36)A0(36); would be open at high m(H0)
            "35:2:meMode = 100",
            "35:3:meMode = 100",
            "35:4:meMode = 100",
            "35:5:meMode = 100",
            "35:7:meMode = 100",
            "35:8:meMode = 100",
#            "35:9:meMode = 100",
            "35:10:meMode= 100",
            "35:12:meMode= 100",
            "35:13:meMode= 100",
            "35:15:meMode= 100",
            "35:18:meMode= 100",
            "35:20:meMode= 100",
 
# Enable H0-->X(mumu)X(mumu) decay
            "35:addChannel = 1 1. 100 6000113 6000113",# Enable H0-->X(mumu)X(jetjet) decay
#            "35:addChannel = 1 1. 100 6000111 6000113",
            "35:onMode = off",
#`           "35:onIfAny = 6000111 6000113"
            "35:onIfAny = 6000113 6000113"
        ),
 
        parameterSets = cms.vstring(
            'pythia8CommonSettings',
#            'pythia8CUEP8M1Settings', #Old Pythia Tune
            'pythia8CP5Settings',
            'pythia8PSweightsSettings',
            'processParameters'
        )
    )
)
 
ProductionFilterSequence = cms.Sequence(generator) 

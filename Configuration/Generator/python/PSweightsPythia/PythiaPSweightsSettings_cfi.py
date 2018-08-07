import FWCore.ParameterSet.Config as cms

pythia8PSweightsSettingsBlock = cms.PSet(
    pythia8PSweightsSettings = cms.vstring(
        'UncertaintyBands:doVariations = on',
# 3 sets of variations for ISR&FSR up/down
# Reduced sqrt(2)/(1/sqrt(2)), Default 2/0.5 and Conservative 4/0.25 variations
# 32 decorrelated variations of muR and non-singular terms (cNS) for each branching type
        'UncertaintyBands:List = {\
isrRedHi isr:muRfac=0.707,fsrRedHi fsr:muRfac=0.707,isrRedLo isr:muRfac=1.414,fsrRedLo fsr:muRfac=1.414,\
isrDefHi isr:muRfac=0.5,fsrDefHi fsr:muRfac=0.5,isrDefLo isr:muRfac=2.0,fsrDefLo fsr:muRfac=2.0,\
isrConHi isr:muRfac=0.25,fsrConHi fsr:muRfac=0.25,isrConLo isr:muRfac=4.0,fsrConLo fsr:muRfac=4.0,\
fsr_G2GG_muR_dn fsr:G2GG:muRfac=0.5,\
fsr_G2GG_muR_up fsr:G2GG:muRfac=2.0,\
fsr_G2QQ_muR_dn fsr:G2QQ:muRfac=0.5,\
fsr_G2QQ_muR_up fsr:G2QQ:muRfac=2.0,\
fsr_Q2QG_muR_dn fsr:Q2QG:muRfac=0.5,\
fsr_Q2QG_muR_up fsr:Q2QG:muRfac=2.0,\
fsr_X2XG_muR_dn fsr:X2XG:muRfac=0.5,\
fsr_X2XG_muR_up fsr:X2XG:muRfac=2.0,\
fsr_G2GG_cNS_dn fsr:G2GG:cNS=-2.0,\
fsr_G2GG_cNS_up fsr:G2GG:cNS=2.0,\
fsr_G2QQ_cNS_dn fsr:G2QQ:cNS=-2.0,\
fsr_G2QQ_cNS_up fsr:G2QQ:cNS=2.0,\
fsr_Q2QG_cNS_dn fsr:Q2QG:cNS=-2.0,\
fsr_Q2QG_cNS_up fsr:Q2QG:cNS=2.0,\
fsr_X2XG_cNS_dn fsr:X2XG:cNS=-2.0,\
fsr_X2XG_cNS_up fsr:X2XG:cNS=2.0\
isr_G2GG_muR_dn isr:G2GG:muRfac=0.5,\
isr_G2GG_muR_up isr:G2GG:muRfac=2.0,\
isr_G2QQ_muR_dn isr:G2QQ:muRfac=0.5,\
isr_G2QQ_muR_up isr:G2QQ:muRfac=2.0,\
isr_Q2QG_muR_dn isr:Q2QG:muRfac=0.5,\
isr_Q2QG_muR_up isr:Q2QG:muRfac=2.0,\
isr_X2XG_muR_dn isr:X2XG:muRfac=0.5,\
isr_X2XG_muR_up isr:X2XG:muRfac=2.0,\
isr_G2GG_cNS_dn isr:G2GG:cNS=-2.0,\
isr_G2GG_cNS_up isr:G2GG:cNS=2.0,\
isr_G2QQ_cNS_dn isr:G2QQ:cNS=-2.0,\
isr_G2QQ_cNS_up isr:G2QQ:cNS=2.0,\
isr_Q2QG_cNS_dn isr:Q2QG:cNS=-2.0,\
isr_Q2QG_cNS_up isr:Q2QG:cNS=2.0,\
isr_X2XG_cNS_dn isr:X2XG:cNS=-2.0,\
isr_X2XG_cNS_up isr:X2XG:cNS=2.0}',
        
        'UncertaintyBands:nFlavQ = 4', # define X=bottom/top in X2XG variations
        'UncertaintyBands:MPIshowers = on',
        'UncertaintyBands:overSampleFSR = 10.0',
        'UncertaintyBands:overSampleISR = 10.0',
        'UncertaintyBands:FSRpTmin2Fac = 20',
        'UncertaintyBands:ISRpTmin2Fac = 1'
        )
)


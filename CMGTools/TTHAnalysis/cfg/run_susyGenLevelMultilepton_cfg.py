##########################################################
##       CONFIGURATION FOR SUSY MULTILEPTON TREES       ##
## skim condition: >= 2 loose leptons, no pt cuts or id ##
##########################################################

import CMGTools.RootTools.fwlite.Config as cfg
from CMGTools.RootTools.fwlite.Config import printComps
from CMGTools.RootTools.RootTools import *

#Load all analyzers
from CMGTools.TTHAnalysis.analyzers.susyCore_modules_cff import * 

# --- LEPTON SKIMMING ---
ttHLepSkim.minLeptons = 2
ttHLepSkim.maxLeptons = 999

ttHGenLevel = cfg.Analyzer(
    'ttHGenLevelOnlyStudy',
    muon_pt_min = 5.,
    electron_pt_min = 7.,
)


from CMGTools.RootTools.samples.samples_8TeV_v517 import triggers_mumu, triggers_ee, triggers_mue, triggers_1mu
# Tree Producer
treeProducer = cfg.Analyzer(
    'treeProducerSusyGenLevelOnly',
    vectorTree = True,
    saveTLorentzVectors = False,  # can set to True to get also the TLorentzVectors, but trees will be bigger
    PDFWeights = PDFWeights,
    # triggerBits = {} # no HLT
    )


#-------- SAMPLES AND TRIGGERS -----------
from CMGTools.RootTools.samples.samples_8TeV_v517 import * 
Test  = kreator.makePrivateMCComponent('Test', '/store/cmst3/user/gpetrucc/maiani', ['m100_g050_3mu.GEN.root'] )
WZ3l_ascms = kreator.makePrivateMCComponent('WZ3l_ascms', '/store/cmst3/user/gpetrucc/maiani/tests', ['xs_wz_3l_ascms.GEN.root'])
WZ3mu_ascms = kreator.makePrivateMCComponent('WZ3mu_ascms', '/store/cmst3/user/gpetrucc/maiani/tests', ['xs_wz_3mu_ascms.GEN.root'])
WZ3mu_offshell = kreator.makePrivateMCComponent('WZ3mu_offshell', '/store/cmst3/user/gpetrucc/maiani/tests', ['xs_wz_3mu_offshell.GEN.root'])

GEN_S3m_lo = kreator.makePrivateMCComponent('GEN_S3m_lo', '/store/cmst3/user/gpetrucc/lmutau/madtests/', [ 'lmutau_lo_test.GEN.root' ])
GEN_S3m_lo_012j = kreator.makePrivateMCComponent('GEN_S3m_lo_012j', '/store/cmst3/user/gpetrucc/lmutau/madtests/', [ 'lmutau_lo_012j_test.GEN.root' ])
GEN_S3m_lo_01j = kreator.makePrivateMCComponent('GEN_S3m_lo_01j', '/store/cmst3/user/gpetrucc/lmutau/madtests/', [ 'lmutau_lo_01j_test.GEN.root' ])
GEN_S3m_nlo = kreator.makePrivateMCComponent('GEN_S3m_nlo', '/store/cmst3/user/gpetrucc/lmutau/madtests/', [ 'lmutau_nlo_test.GEN.root' ])
GEN_S3m_nlo_01j = kreator.makePrivateMCComponent('GEN_S3m_nlo_01j', '/store/cmst3/user/gpetrucc/lmutau/madtests/', [ 'lmutau_nlo_01j_test.GEN.root', 'lmutau_nlo_01j_test.2.GEN.root' ])
GEN_S3m_lo_direct = kreator.makePrivateMCComponent('GEN_S3m_lo_direct', '/store/cmst3/user/gpetrucc/lmutau/madtests/', [ 'lmutau_lo_direct_test.GEN.root' ])

GEN_Bonly_S3m = kreator.makePrivateMCComponent('GEN_Bonly_S3m', '/store/cmst3/user/gpetrucc/lmutau/', [ 'Bonly_S3m_m100_g050_8TeV.GEN.root' ])
GEN_SBI_S3m_m100_g050 = kreator.makePrivateMCComponent('GEN_SBI_S3m_m100_g050', '/store/cmst3/user/gpetrucc/lmutau/', [ 'SBI_S3m_m100_g050_8TeV.GEN.root' ])
GEN_SBI_S3m_m105_g052 = kreator.makePrivateMCComponent('GEN_SBI_S3m_m105_g052', '/store/cmst3/user/gpetrucc/lmutau/', [ 'SBI_S3m_m105_g052_8TeV.GEN.root' ])
GEN_SBI_S3m_m110_g055 = kreator.makePrivateMCComponent('GEN_SBI_S3m_m110_g055', '/store/cmst3/user/gpetrucc/lmutau/', [ 'SBI_S3m_m110_g055_8TeV.GEN.root' ])
GEN_SBI_S3m_m115_g057 = kreator.makePrivateMCComponent('GEN_SBI_S3m_m115_g057', '/store/cmst3/user/gpetrucc/lmutau/', [ 'SBI_S3m_m115_g057_8TeV.GEN.root' ])
GEN_SBI_S3m_m120_g060 = kreator.makePrivateMCComponent('GEN_SBI_S3m_m120_g060', '/store/cmst3/user/gpetrucc/lmutau/', [ 'SBI_S3m_m120_g060_8TeV.GEN.root' ])
GEN_SBI_S3m_m125_g062 = kreator.makePrivateMCComponent('GEN_SBI_S3m_m125_g062', '/store/cmst3/user/gpetrucc/lmutau/', [ 'SBI_S3m_m125_g062_8TeV.GEN.root' ])
GEN_SBI_S3m_m70_g035 = kreator.makePrivateMCComponent('GEN_SBI_S3m_m70_g035', '/store/cmst3/user/gpetrucc/lmutau/', [ 'SBI_S3m_m70_g035_8TeV.GEN.root' ])
GEN_SBI_S3m_m75_g037 = kreator.makePrivateMCComponent('GEN_SBI_S3m_m75_g037', '/store/cmst3/user/gpetrucc/lmutau/', [ 'SBI_S3m_m75_g037_8TeV.GEN.root' ])
GEN_SBI_S3m_m80_g040 = kreator.makePrivateMCComponent('GEN_SBI_S3m_m80_g040', '/store/cmst3/user/gpetrucc/lmutau/', [ 'SBI_S3m_m80_g040_8TeV.GEN.root' ])
GEN_SBI_S3m_m85_g042 = kreator.makePrivateMCComponent('GEN_SBI_S3m_m85_g042', '/store/cmst3/user/gpetrucc/lmutau/', [ 'SBI_S3m_m85_g042_8TeV.GEN.root' ])
GEN_SBI_S3m_m90_g045 = kreator.makePrivateMCComponent('GEN_SBI_S3m_m90_g045', '/store/cmst3/user/gpetrucc/lmutau/', [ 'SBI_S3m_m90_g045_8TeV.GEN.root' ])
GEN_SBI_S3m_m95_g047 = kreator.makePrivateMCComponent('GEN_SBI_S3m_m95_g047', '/store/cmst3/user/gpetrucc/lmutau/', [ 'SBI_S3m_m95_g047_8TeV.GEN.root' ])
GEN_Sonly_S3m_m100_g050 = kreator.makePrivateMCComponent('GEN_Sonly_S3m_m100_g050', '/store/cmst3/user/gpetrucc/lmutau/v1/', [ 'S3m_m100_g050_8TeV.GEN.root' ])
GEN_Sonly_S3m_m105_g052 = kreator.makePrivateMCComponent('GEN_Sonly_S3m_m105_g052', '/store/cmst3/user/gpetrucc/lmutau/v1/', [ 'S3m_m105_g052_8TeV.GEN.root' ])
GEN_Sonly_S3m_m110_g055 = kreator.makePrivateMCComponent('GEN_Sonly_S3m_m110_g055', '/store/cmst3/user/gpetrucc/lmutau/v1/', [ 'S3m_m110_g055_8TeV.GEN.root' ])
GEN_Sonly_S3m_m115_g057 = kreator.makePrivateMCComponent('GEN_Sonly_S3m_m115_g057', '/store/cmst3/user/gpetrucc/lmutau/v1/', [ 'S3m_m115_g057_8TeV.GEN.root' ])
GEN_Sonly_S3m_m120_g060 = kreator.makePrivateMCComponent('GEN_Sonly_S3m_m120_g060', '/store/cmst3/user/gpetrucc/lmutau/v1/', [ 'S3m_m120_g060_8TeV.GEN.root' ])
GEN_Sonly_S3m_m125_g062 = kreator.makePrivateMCComponent('GEN_Sonly_S3m_m125_g062', '/store/cmst3/user/gpetrucc/lmutau/v1/', [ 'S3m_m125_g062_8TeV.GEN.root' ])
GEN_Sonly_S3m_m70_g035 = kreator.makePrivateMCComponent('GEN_Sonly_S3m_m70_g035', '/store/cmst3/user/gpetrucc/lmutau/v1/', [ 'S3m_m70_g035_8TeV.GEN.root' ])
GEN_Sonly_S3m_m75_g037 = kreator.makePrivateMCComponent('GEN_Sonly_S3m_m75_g037', '/store/cmst3/user/gpetrucc/lmutau/v1/', [ 'S3m_m75_g037_8TeV.GEN.root' ])
GEN_Sonly_S3m_m80_g040 = kreator.makePrivateMCComponent('GEN_Sonly_S3m_m80_g040', '/store/cmst3/user/gpetrucc/lmutau/v1/', [ 'S3m_m80_g040_8TeV.GEN.root' ])
GEN_Sonly_S3m_m85_g042 = kreator.makePrivateMCComponent('GEN_Sonly_S3m_m85_g042', '/store/cmst3/user/gpetrucc/lmutau/v1/', [ 'S3m_m85_g042_8TeV.GEN.root' ])
GEN_Sonly_S3m_m90_g045 = kreator.makePrivateMCComponent('GEN_Sonly_S3m_m90_g045', '/store/cmst3/user/gpetrucc/lmutau/v1/', [ 'S3m_m90_g045_8TeV.GEN.root' ])
GEN_Sonly_S3m_m95_g047 = kreator.makePrivateMCComponent('GEN_Sonly_S3m_m95_g047', '/store/cmst3/user/gpetrucc/lmutau/v1/', [ 'S3m_m95_g047_8TeV.GEN.root' ])


### ====== SUSY: PRIVATE PRODUCTIONS ==========
GEN_T2tt_py8had = kreator.makePrivateMCComponent('GEN_T2tt_py8had', '/store/cmst3/user/gpetrucc/SUSY/TestProd/T2tt/', [ 'T2tt_onshell_pyt8had.root' ] + [ "T2tt_onshell_pyt8had.run_0%d_chunk_%d.root" % (i,j) for i in 2,3,4,5, for j in 0,1,2,3,4 ] )
GEN_T2tt_py8dec_py8had = kreator.makePrivateMCComponent('GEN_T2tt_py8dec_py8had', '/store/cmst3/user/gpetrucc/SUSY/TestProd/T2tt/', [ 'T2tt_onshell_py8decay_pyt8had.root' ] + [ "T2tt_onshell_py8decay_pyt8had.run_0%d_chunk_%d.root" % (i,j) for i in 2,3,4,5, for j in 0,1,2,3,4 ])
GEN_T2tt_mgdec_py8had  = kreator.makePrivateMCComponent('GEN_T2tt_mgdec_py8had',  '/store/cmst3/user/gpetrucc/SUSY/TestProd/T2tt/', [ 'T2tt_decayed_pyt8had.root' ] + [ "T2tt_decayed_pyt8had.run_0%d_chunk_%d.root" % (i,j) for i in 3,4,5,6, for j in 0,1,2,3,4 ])

GEN_T2tt_py8had_ch = kreator.makePrivateMCComponent('GEN_T2tt_py8had_ch', '/store/cmst3/user/gpetrucc/SUSY/TestProd/T2tt/', [ 'T2tt_onshell_pyt8had_chargino.root' ]+[ "T2tt_onshell_pyt8had_chargino.run_0%d_chunk_0%d.root" % (i,j) for i in 2,3,4,5, for j in 0,1,2,3,4 ] ) 
GEN_T2tt_mgdec_py8had_ch  = kreator.makePrivateMCComponent('GEN_T2tt_mgdec_py8had_ch',  '/store/cmst3/user/gpetrucc/SUSY/TestProd/T2tt/', [ "T2tt_decayed_pyt8had_chargino.run_0%d_chunk_0%d.root" % (i,j) for i in (1,) for j in 0,1,2,3,4 ])
GEN_T2tt_mgdec_py8had_both  = kreator.makePrivateMCComponent('GEN_T2tt_mgdec_py8had_both',  '/store/cmst3/user/gpetrucc/SUSY/TestProd/T2tt/', [ "T2tt_decayed_pyt8had_both.run_0%d_chunk_0%d.root" % (i,j) for i in 1,2 for j in 0,1,2,3,4 ])

GEN_T1tttt_mGo800_mStop300_mChi280_mg5 = kreator.makePrivateMCComponent('GEN_T1tttt_mGo800_mStop300_mChi280_mg5', '/store/cmst3/user/gpetrucc/SUSY/Prod/T1tttt_mGo800_mStop300_mChi280_mg5dec_pythia8/', [ "T1tttt_mGo800_mStop300_mChi280_mg5dec_pythia8.run_%02d_chunk_%02d.root" % (i,j) for i in (1,2) for j in xrange(10) ])
#GEN_T1tttt_mGo1300_mStop300_mChi280_mg5 = kreator.makePrivateMCComponent('GEN_T1tttt_mGo1300_mStop300_mChi280_mg5', '/store/cmst3/user/gpetrucc/SUSY/Prod/T1tttt_mGo1300_mStop300_mChi280_mg5dec_pythia8/', [ "T1tttt_mGo1300_mStop300_mChi280_mg5dec_pythia8.run_%02d_chunk_%02d.root" % (i,j) for i in (1,2) for j in xrange(10) ])
GEN_T1tttt_mGo800_mStop300_mCh285_mChi280_mg5 = kreator.makePrivateMCComponent('GEN_T1tttt_mGo800_mStop300_mCh285_mChi280_mg5', '/store/cmst3/user/gpetrucc/SUSY/Prod/T1tttt_mGo800_mStop300_mCh285_mChi280_mg5dec_pythia8/', [ "T1tttt_mGo800_mStop300_mCh285_mChi280_mg5dec_pythia8.run_%02d_chunk_%d.root" % (i,j) for i in (1,) for j in xrange(20) ])
#GEN_T1tttt_mGo1300_mStop300_mCh285_mChi280_mg5 = kreator.makePrivateMCComponent('GEN_T1tttt_mGo1300_mStop300_mCh285_mChi280_mg5', '/store/cmst3/user/gpetrucc/SUSY/Prod/T1tttt_mGo1300_mStop300_mCh285_mChi280_mg5dec_pythia8/', [ "T1tttt_mGo1300_mStop300_mCh285_mChi280_mg5dec_pythia8.run_%02d_chunk_%02d.root" % (i,j) for i in (1,2) for j in xrange(10) ])

### ====== SUSY: PRIVATE DECAY+HADRONIZATION OF CENTRALLY PRODUCED LHE FILES ==========
eosGenFiles = [ x.strip() for x in open(os.environ["CMSSW_BASE"]+"/src/CMGTools/RootTools/python/samples/genLevel-susySMS-13TeV", "r") ]
print eosGenFiles
def mkGen(name,what):
    return kreator.makePrivateMCComponent(name, '/'+what, [ f for f in eosGenFiles if ("/%s/"%what) in f ] )
GEN_T1tttt_2J_mGo800_mStop300_mChi280_py8 = mkGen('GEN_T1tttt_2J_mGo800_mStop300_mChi280_py8', 'T1ttt_2J_mGo800_mStop300_mChi280_pythia8-4bodydec')
GEN_T1tttt_2J_mGo1300_mStop300_mChi280_py8 = mkGen('GEN_T1tttt_2J_mGo1300_mStop300_mChi280_py8', 'T1ttt_2J_mGo1300_mStop300_mChi280_pythia8-4bodydec')
GEN_T1tttt_2J_mGo800_mStop300_mCh285_mChi280_py8 = mkGen('GEN_T1tttt_2J_mGo800_mStop300_mCh285_mChi280_py8', 'T1ttt_2J_mGo800_mStop300_mCh285_mChi280_pythia8-23bodydec')
GEN_T1tttt_2J_mGo1300_mStop300_mCh285_mChi280_py8 = mkGen('GEN_T1tttt_2J_mGo1300_mStop300_mCh285_mChi280_py8', 'T1ttt_2J_mGo1300_mStop300_mCh285_mChi280_pythia8-23bodydec')
GEN_T1tttt_2J_mGo1300_mStop300_mCh285_mChi280_py8_dilep = mkGen('GEN_T1tttt_2J_mGo1300_mStop300_mCh285_mChi280_py8_dilep', 'T1tttt_2J_mGo1300_mStop300_mCh285_mChi280_23bodydec_dilepfilter')


#selectedComponents = [ GEN_S3m_lo, GEN_S3m_lo_012j, GEN_S3m_lo_01j, GEN_S3m_nlo, GEN_S3m_nlo_01j, GEN_S3m_lo_direct ]
selectedComponents = [ GEN_T2tt_py8had, GEN_T2tt_py8dec_py8had, GEN_T2tt_mgdec_py8had ]
#selectedComponents = [ GEN_Bonly_S3m, GEN_SBI_S3m_m100_g050, GEN_SBI_S3m_m105_g052, GEN_SBI_S3m_m110_g055, GEN_SBI_S3m_m115_g057, GEN_SBI_S3m_m120_g060, GEN_SBI_S3m_m125_g062, GEN_SBI_S3m_m70_g035, GEN_SBI_S3m_m75_g037, GEN_SBI_S3m_m80_g040, GEN_SBI_S3m_m85_g042, GEN_SBI_S3m_m90_g045, GEN_SBI_S3m_m95_g047, GEN_Sonly_S3m_m100_g050, GEN_Sonly_S3m_m105_g052, GEN_Sonly_S3m_m110_g055, GEN_Sonly_S3m_m115_g057, GEN_Sonly_S3m_m120_g060, GEN_Sonly_S3m_m125_g062, GEN_Sonly_S3m_m70_g035, GEN_Sonly_S3m_m75_g037, GEN_Sonly_S3m_m80_g040, GEN_Sonly_S3m_m85_g042, GEN_Sonly_S3m_m90_g045, GEN_Sonly_S3m_m95_g047 ]
#selectedComponents = [ GEN_Bonly_S3m, GEN_SBI_S3m_m110_g055, GEN_Sonly_S3m_m110_g055 ]
selectedComponents = [ GEN_T1tttt_2J_mGo800_mStop300_mChi280_py8, GEN_T1tttt_2J_mGo1300_mStop300_mChi280_py8, GEN_T1tttt_2J_mGo800_mStop300_mCh285_mChi280_py8, GEN_T1tttt_2J_mGo1300_mStop300_mCh285_mChi280_py8 ]
#selectedComponents = [ GEN_T2tt_mgdec_py8had_ch, GEN_T2tt_mgdec_py8had_both, GEN_T2tt_py8had_ch ]
#selectedComponents = [ GEN_T2tt_py8had, GEN_T2tt_mgdec_py8had ]
selectedComponents = [ GEN_T1tttt_2J_mGo1300_mStop300_mCh285_mChi280_py8, GEN_T1tttt_2J_mGo1300_mStop300_mCh285_mChi280_py8_dilep ]

for c in selectedComponents: c.splitFactor = 100
#-------- SEQUENCE

sequence = cfg.Sequence([
    skimAnalyzer,
    ttHGenLevel, 
    ttHLepSkim,
    treeProducer,
    ])


#-------- HOW TO RUN
test = 0
if test==1:
    # test a single component, using a single thread.
    comp = GEN_T2tt_mgdec_py8had_both
    comp.files = comp.files[:1]
    selectedComponents = [comp]
    comp.splitFactor = 1
elif test==2:    
    # test all components (1 thread per component).
    for comp in selectedComponents:
        comp.splitFactor = 1
        comp.files = comp.files[:1]



config = cfg.Config( components = selectedComponents,
                     sequence = sequence )

printComps(config.components, True)

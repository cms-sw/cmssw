##########################################################
##       CONFIGURATION FOR SUSY MULTILEPTON TREES       ##
## skim condition: >= 2 loose leptons, no pt cuts or id ##
##########################################################
import PhysicsTools.HeppyCore.framework.config as cfg


#-------- LOAD ALL ANALYZERS -----------

from CMGTools.TTHAnalysis.analyzers.susyCore_modules_cff import *


#-------- REDEFINE WHAT I NEED -----------

# Lepton Skimming
ttHLepSkim.minLeptons = 2
ttHLepSkim.maxLeptons = 999
#ttHLepSkim.idCut  = ""
#ttHLepSkim.ptCuts = []

# Run miniIso
lepAna.doMiniIsolation = True
lepAna.packedCandidates = 'packedPFCandidates'
lepAna.miniIsolationPUCorr = 'rhoArea'
lepAna.miniIsolationVetoLeptons = None # use 'inclusive' to veto inclusive leptons and their footprint in all isolation cones
    
# Lepton Preselection
lepAna.loose_electron_id = "POG_MVA_ID_Run2_NonTrig_VLoose"

isolation = "miniIso"
#isolation = "ptRel"
if isolation == "ptRel": 
    # delay isolation cut for leptons of pt > 10, for which we do pTrel recovery
    lepAna.loose_muon_isoCut     = lambda muon : muon.relIso03 < 0.5 or muon.pt() > 10
    lepAna.loose_electron_isoCut = lambda elec : elec.relIso03 < 0.5 or elec.pt() > 10
    # in the cleaning, keep the jet if the lepton fails relIso or ptRel
    jetAna.jetLepArbitration = lambda jet,lepton : (
        lepton if (lepton.relIso03 < 0.4 or ptRelv1(lepton.p4(),jet.p4()) > 5) else jet
    )
    ttHCoreEventAna.leptonMVAKindTTH = "SusyWithBoost"
    ttHCoreEventAna.leptonMVAKindSusy = "SusyWithBoost" 
    ttHCoreEventAna.leptonMVAPathTTH = "CMGTools/TTHAnalysis/macros/leptons/trainingPHYS14leptonMVA_PHYS14eleMVA_MiniIso_ttH/weights/%s_BDTG.weights.xml"
    ttHCoreEventAna.leptonMVAPathSusy = "CMGTools/TTHAnalysis/macros/leptons/trainingPHYS14leptonMVA_PHYS14eleMVA_MiniIso_SusyT1/weights/%s_BDTG.weights.xml"
    # insert a second skimmer after the jet cleaning 
    ttHLepSkim2 = cfg.Analyzer(
        ttHLepSkimmer, name='ttHLepSkimmer2',
        minLeptons = 2,
        maxLeptons = 999,
        )
    susyCoreSequence.insert(susyCoreSequence.index(jetAna)+1, ttHLepSkim2)
elif isolation == "miniIso": 
    lepAna.loose_muon_isoCut     = lambda muon : muon.miniRelIso < 0.4
    lepAna.loose_electron_isoCut = lambda elec : elec.miniRelIso < 0.4
elif isolation == None:
    lepAna.loose_muon_isoCut     = lambda muon : True
    lepAna.loose_electron_isoCut = lambda elec : True
else:
    # nothing to do, will use normal relIso03
    pass

# Switch off slow photon MC matching
photonAna.do_mc_match = False


#-------- ADDITIONAL ANALYZERS -----------

## Event Analyzer for susy multi-lepton (at the moment, it's the TTH one)
from CMGTools.TTHAnalysis.analyzers.ttHLepEventAnalyzer import ttHLepEventAnalyzer
ttHEventAna = cfg.Analyzer(
    ttHLepEventAnalyzer, name="ttHLepEventAnalyzer",
    minJets25 = 0,
    )

## JetTau analyzer, to be called (for the moment) once bjetsMedium are produced
from CMGTools.TTHAnalysis.analyzers.ttHJetTauAnalyzer import ttHJetTauAnalyzer
ttHJetTauAna = cfg.Analyzer(
    ttHJetTauAnalyzer, name="ttHJetTauAnalyzer",
    )

## Insert the FatJet, SV, HeavyFlavour analyzers in the sequence
susyCoreSequence.insert(susyCoreSequence.index(ttHCoreEventAna), 
                        ttHFatJetAna)
susyCoreSequence.insert(susyCoreSequence.index(ttHCoreEventAna), 
                        ttHSVAna)
susyCoreSequence.insert(susyCoreSequence.index(ttHCoreEventAna), 
                        ttHHeavyFlavourHadronAna)

## Insert declustering analyzer
from CMGTools.TTHAnalysis.analyzers.ttHDeclusterJetsAnalyzer import ttHDeclusterJetsAnalyzer
ttHDecluster = cfg.Analyzer(
    ttHDeclusterJetsAnalyzer, name='ttHDecluster',
    lepCut     = lambda lep,ptrel : lep.pt() > 10,
    maxSubjets = 6, # for exclusive reclustering
    ptMinSubjets = 5, # for inclusive reclustering
    drMin      = 0.2, # minimal deltaR(l,subjet) required for a successful subjet match
    ptRatioMax = 1.5, # maximum pt(l)/pt(subjet) required for a successful match
    ptRatioDiff = 0.1,  # cut on abs( pt(l)/pt(subjet) - 1 ) sufficient to call a match successful
    drMatch     = 0.02, # deltaR(l,subjet) sufficient to call a match successful
    ptRelMin    = 5,    # maximum ptRelV1(l,subjet) sufficient to call a match successful
    prune       = True, # also do pruning of the jets 
    pruneZCut       = 0.1, # pruning parameters (usual value in CMS: 0.1)
    pruneRCutFactor = 0.5, # pruning parameters (usual value in CMS: 0.5)
    verbose     = 0,   # print out the first N leptons
    jetCut = lambda jet : jet.pt() > 20,
    mcPartonPtCut = 20,
    mcLeptonPtCut =  5,
    mcTauPtCut    = 15,
    )
susyCoreSequence.insert(susyCoreSequence.index(ttHFatJetAna)+1, ttHDecluster)


from CMGTools.TTHAnalysis.analyzers.treeProducerSusyMultilepton import * 
## Tree Producer
treeProducer = cfg.Analyzer(
     AutoFillTreeProducer, name='treeProducerSusyMultilepton',
     vectorTree = True,
     saveTLorentzVectors = False,  # can set to True to get also the TLorentzVectors, but trees will be bigger
     defaultFloatType = 'F', # use Float_t for floating point
     PDFWeights = PDFWeights,
     globalVariables = susyMultilepton_globalVariables,
     globalObjects = susyMultilepton_globalObjects,
     collections = susyMultilepton_collections,
)

## histo counter
susyCoreSequence.insert(susyCoreSequence.index(skimAnalyzer),
                        susyCounter)


#-------- SAMPLES AND TRIGGERS -----------


from CMGTools.RootTools.samples.triggers_13TeV_Spring15 import *
from CMGTools.RootTools.samples.triggers_8TeV import triggers_1mu_8TeV, triggers_mumu_8TeV, triggers_mue_8TeV, triggers_ee_8TeV;
triggerFlagsAna.triggerBits = {
    'DoubleMu' : triggers_mumu_iso,
    'DoubleMuSS' : triggers_mumu_ss,
    'DoubleMuNoIso' : triggers_mumu_noniso,
    'DoubleEl' : triggers_ee,
    'MuEG'     : triggers_mue,
    'DoubleMuHT' : triggers_mumu_ht,
    'DoubleElHT' : triggers_ee_ht,
    'MuEGHT' : triggers_mue_ht,
    'TripleEl' : triggers_3e,
    'TripleMu' : triggers_3mu,
    'TripleMuA' : triggers_3mu_alt,
    'DoubleMuEl' : triggers_2mu1e,
    'DoubleElMu' : triggers_2e1mu,
    'SingleMu' : triggers_1mu_iso,
    'SingleMu50ns' : triggers_1mu_iso_50ns,
    'SingleEl'     : triggers_1e,
    'SingleEl50ns' : triggers_1e_50ns,
    'SingleMu_8TeV' : triggers_1mu_8TeV + triggers_1mu_iso_r,
    'DoubleMu_8TeV' : triggers_mumu_8TeV + triggers_mumu_run1,
    'MuEG_8TeV'     : triggers_mue_8TeV + triggers_mue_run1,
    'DoubleEl_8TeV' : triggers_ee_8TeV + triggers_ee_run1,
}

from CMGTools.RootTools.samples.samples_13TeV_74X import *
from CMGTools.RootTools.samples.samples_13TeV_74X_susySignalsPriv import *
from CMGTools.RootTools.samples.samples_8TeVReReco_74X import *

selectedComponents = [ SingleMu_742, MuEG_742, DoubleMu_742 ] 
selectedComponents = [ TTJets, TTJets_LO, WJetsToLNu, DYJetsToLL_M10to50,  DYJetsToLL_M50,  ] + SingleTop + DiBosons
selectedComponents = mcSamplesPriv 

if False: # select only a subset of a sample, corresponding to a given luminosity (assuming ~30k events per MiniAOD file, which is ok for central production)
    target_lumi = 5000 # in inverse picobarns
    for c in selectedComponents:
        nfiles = int(min(ceil(target_lumi * c.xSection / 30e3), len(c.files)))
        print "For component %s, will want %d/%d files" % (c.name, nfiles, len(c.files))
        c.files = c.files[:nfiles]
        c.splitFactor = len(c.files)

#  selectedComponents = [
#    ] + WJetsToLNuHT + DYJetsM50HT + [ #DYJetsToLL_M50,
#     TTJets ]+ SingleTop +[
#     TTWJets,TTZJets, TTH,
#     WZJetsTo3LNu, ZZTo4L,
#     #GGHZZ4L, GGHTT, VBFTT, 
#     SMS_T1tttt_2J_mGl1500_mLSP100, SMS_T1tttt_2J_mGl1200_mLSP800,
#     T5ttttDeg_mGo1000_mStop300_mCh285_mChi280, T5ttttDeg_mGo1000_mStop300_mCh285_mChi280_dil,
#     T5qqqqWW_mGo1200_mCh1000_mChi800_dilep, T5qqqqWWDeg_mGo1000_mCh315_mChi300_dilep
#  ]
#  if False:
#      ttHLepSkim.minLeptons = 1
#      QCDPtEMEnriched.remove(QCD_Pt10to20_EMEnriched)
#      selectedComponents = [ QCD_Mu15 ] + QCD_Mu5 + QCDPtEMEnriched + QCDPtbcToE


# -- fine splitting, for some private MC samples with a single file
#for comp in selectedComponents:
#    comp.splitFactor = 1
#    comp.fineSplitFactor = 40

    
#-------- SEQUENCE -----------

sequence = cfg.Sequence(susyCoreSequence+[
        ttHJetTauAna,
        ttHEventAna,
        treeProducer,
    ])
preprocessor = None

#-------- HOW TO RUN -----------

from PhysicsTools.HeppyCore.framework.heppy import getHeppyOption
test = getHeppyOption('test')
if test == '1':
    comp = TTJets
    comp.files = comp.files[:1]
    comp.splitFactor = 1
    if not getHeppyOption('single'):
        comp.fineSplitFactor = 4
    selectedComponents = [ comp ]
elif test == '2':
    for comp in selectedComponents:
        comp.files = comp.files[:1]
        comp.splitFactor = 1
        comp.fineSplitFactor = 1
elif test == 'SingleMu':
    comp = SingleMu
    comp.files = comp.files[:1]
    comp.splitFactor = 1
    selectedComponents = [ comp ]
elif test == '3':
    for comp in selectedComponents:
        comp.files = comp.files[:1]
        comp.splitFactor = 1
        comp.fineSplitFactor = 4
elif test == '5':
    for comp in selectedComponents:
        comp.files = comp.files[:5]
        comp.splitFactor = 1
        comp.fineSplitFactor = 5
elif test == '2lss-sync': # sync
    #eventSelector.toSelect = [ 11809 ]
    #sequence = cfg.Sequence([eventSelector] + susyCoreSequence+[ ttHEventAna, treeProducer, ])
    jetAna.recalibrateJets = False 
    jetAna.smearJets       = False 
    comp = SMS_T1tttt_2J_mGl1200_mLSP800
    comp.files = [ 'root://eoscms//eos/cms/store/mc/Phys14DR/SMS-T1tttt_2J_mGl-1200_mLSP-800_Tune4C_13TeV-madgraph-tauola/MINIAODSIM/PU20bx25_tsg_PHYS14_25_V1-v1/00000/0CD15D7F-4E6B-E411-AEB4-002590DB9216.root' ]
    comp.splitFactor = 1
    comp.fineSplitFactor = 10
    selectedComponents = [ comp ]
elif test == '74X-MC':
    what = getHeppyOption("sample")
    if what == "TT":
        ttHLepSkim.minLeptons = 0
        selectedComponents = [ TT_bx25 ]
    elif what == "Z":
        selectedComponents = [ ZEE_bx25, ZMM_bx25, ZTT_bx25 ]
    else:
        selectedComponents = RelVals740
    if not getHeppyOption("all"):
        for comp in selectedComponents:
            comp.files = comp.files[:1]
            comp.splitFactor = 1
            comp.fineSplitFactor = 1 if getHeppyOption("single") else 4
elif test == '74X-Data':
    from CMGTools.RootTools.samples.samples_13TeV_74X import *
    from CMGTools.RootTools.samples.samples_8TeVReReco_74X import *
    what = getHeppyOption("sample")
    if what == "SingleMu":
        selectedComponents = [ SingleMu_740p9 ]
    elif what == "Z":
        selectedComponents = [ SingleMuZ_740p9, DoubleElectronZ_740p9 ]
    elif what == "MuEG":
        selectedComponents = [ MuEG_740p9 ]
    else:
        selectedComponents = dataSamples740p9
    if not getHeppyOption("all"):
        for comp in selectedComponents:
            comp.files = comp.files[:1]
            comp.splitFactor = 1
            comp.fineSplitFactor = 1 if getHeppyOption("single") else 4
elif test == "express":
    selectedComponents = [ MuEG_740p9 ]
    comp = selectedComponents[0]
    comp.files = [ 'root://eoscms//eos/cms/store/express/Run2015A/ExpressPhysics/FEVT/Express-v1/000/246/908/00000/04B152E7-DE09-E511-8B18-02163E011D4A.root' ]
    comp.name  = 'ExpressPhysics'
    comp.triggers = []
    comp.json     = None
    jetAna.recalibrateJets = False 
    jetAna.smearJets       = False 
    ttHLepSkim.minLeptons = 0
    sequence.remove(jsonAna)
    # preprocessor cfg to be created with
    #    cmsDriver.py miniAOD-data -s PAT --data --runUnscheduled --eventcontent MINIAOD --conditions GR_P_V56 --no_exec
    #    sed -i 's/process.MINIAODoutput_step/process.endpath/' miniAOD-data_PAT.py
    from PhysicsTools.Heppy.utils.cmsswPreprocessor import CmsswPreprocessor
    preprocessor = CmsswPreprocessor("miniAOD-data_PAT.py")

## output histogram
outputService=[]
from PhysicsTools.HeppyCore.framework.services.tfile import TFileService
output_service = cfg.Service(
    TFileService,
    'outputfile',
    name="outputfile",
    fname='treeProducerSusyMultilepton/tree.root',
    option='recreate'
    )    
outputService.append(output_service)

# the following is declared in case this cfg is used in input to the heppy.py script
from PhysicsTools.HeppyCore.framework.eventsfwlite import Events
from CMGTools.TTHAnalysis.tools.EOSEventsWithDownload import EOSEventsWithDownload
event_class = EOSEventsWithDownload
if getHeppyOption("nofetch"):
    event_class = Events 
config = cfg.Config( components = selectedComponents,
                     sequence = sequence,
                     services = outputService, 
                     preprocessor = preprocessor, 
                     events_class = event_class)

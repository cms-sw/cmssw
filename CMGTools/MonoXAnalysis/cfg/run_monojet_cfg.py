##########################################################
##       CONFIGURATION FOR EXO MONOJET TREES            ##
## skim condition:   MET > 200 GeV                      ##
##########################################################
import PhysicsTools.HeppyCore.framework.config as cfg

#Load all analyzers
from CMGTools.MonoXAnalysis.analyzers.dmCore_modules_cff import * 
from PhysicsTools.Heppy.analyzers.objects.METAnalyzer import *

# Redefine what I need

# --- MONOJET SKIMMING ---
monoJetSkim.metCut = 200
monoJetSkim.jetPtCuts = []

# --- W->munu control sample SKIMMING ---
monoJetCtrlLepSkim.minLeptons = 0

# run miniIso
lepAna.doMiniIsolation = True
lepAna.packedCandidates = 'packedPFCandidates'
lepAna.miniIsolationPUCorr = 'rhoArea'
lepAna.miniIsolationVetoLeptons = None # use 'inclusive' to veto inclusive leptons and their footprint in all isolation cones
## will become miniIso perhaps?
#lepAna.loose_muon_isoCut     = lambda muon : muon.relIso03 < 10.5
#lepAna.loose_electron_isoCut = lambda electron : electron.relIso03 < 10.5
    

# switch off slow photon MC matching
photonAna.do_mc_match = False


##------------------------------------------
##  TOLOLOGIAL VARIABLES: RAZOR
##------------------------------------------
from PhysicsTools.Heppy.analyzers.eventtopology.RazorAnalyzer import RazorAnalyzer
monoXRazorAna = cfg.Analyzer(
    RazorAnalyzer, name = 'RazorAnalyzer',
    doOnlyDefault = False
    )

##------------------------------------------
##  TOLOLOGIAL VARIABLES: MT2
##------------------------------------------
from CMGTools.TTHAnalysis.analyzers.ttHTopoVarAnalyzer import ttHTopoVarAnalyzer
ttHTopoJetAna = cfg.Analyzer(
    ttHTopoVarAnalyzer, name = 'ttHTopoVarAnalyzer',
    doOnlyDefault = True
    )

from PhysicsTools.Heppy.analyzers.eventtopology.MT2Analyzer import MT2Analyzer
monoXMT2Ana = cfg.Analyzer(
    MT2Analyzer, name = 'MT2Analyzer',
    doOnlyDefault = False
    )

##------------------------------------------
##  TOLOLOGIAL VARIABLES: ALPHAT
##------------------------------------------
from CMGTools.TTHAnalysis.analyzers.ttHAlphaTVarAnalyzer import ttHAlphaTVarAnalyzer
ttHAlphaTAna = cfg.Analyzer(
    ttHAlphaTVarAnalyzer, name = 'ttHAlphaTVarAnalyzer',
    )

from CMGTools.TTHAnalysis.analyzers.ttHAlphaTControlAnalyzer import ttHAlphaTControlAnalyzer
ttHAlphaTControlAna = cfg.Analyzer(
    ttHAlphaTVarAnalyzer, name = 'ttHAlphaTControlAnalyzer',
    )
##-----------------------------------------------
##  TOLOLOGIAL VARIABLES: MONOJET SPECIFIC ONES
##-----------------------------------------------
from CMGTools.MonoXAnalysis.analyzers.monoJetVarAnalyzer import monoJetVarAnalyzer
monoJetVarAna = cfg.Analyzer(
    monoJetVarAnalyzer, name = 'monoJetVarAnalyzer',
    )

##------------------------------------------
# Event Analyzer for monojet 
##------------------------------------------
from CMGTools.MonoXAnalysis.analyzers.monoJetEventAnalyzer import monoJetEventAnalyzer
MonoJetEventAna = cfg.Analyzer(
    monoJetEventAnalyzer, name="monoJetEventAnalyzer",
    minJets25 = 0,
    )


from CMGTools.MonoXAnalysis.analyzers.treeProducerDarkMatterMonoJet import * 
## Tree Producer
treeProducer = cfg.Analyzer(
     AutoFillTreeProducer, name='treeProducerDarkMatterMonoJet',
     vectorTree = True,
     saveTLorentzVectors = False,  # can set to True to get also the TLorentzVectors, but trees will be bigger
     defaultFloatType = 'F', # use Float_t for floating point
     PDFWeights = PDFWeights,
     doPDFVars = True,
     globalVariables = dmMonoJet_globalVariables,
     globalObjects = dmMonoJet_globalObjects,
     collections = dmMonoJet_collections,
)

## histo counter
# dmCoreSequence.insert(dmCoreSequence.index(skimAnalyzer),
#                       dmCounter)

#-------- SAMPLES AND TRIGGERS -----------
from CMGTools.MonoXAnalysis.samples.samples_monojet_13TeV_PHYS14 import triggers_monojet
triggerFlagsAna.triggerBits = {
    'MonoJet' : triggers_monojet,
}

from CMGTools.MonoXAnalysis.samples.samples_monojet_13TeV_PHYS14 import *

signalSamples = MonojetSignalSamples
backgroundSamples =  WJetsToLNuHT + ZJetsToNuNuHT + SingleTop + [TTJets] + DYJetsM50HT + GJetsHT + QCDHT
selectedComponents = backgroundSamples + signalSamples

#-------- SEQUENCE
sequence = cfg.Sequence(dmCoreSequence+[
   monoXRazorAna,
   monoXMT2Ana,
   ttHFatJetAna,
   ttHAlphaTAna,
   ttHAlphaTControlAna,
   monoJetVarAna,
   MonoJetEventAna,
   treeProducer,
    ])

# -- fine splitting, for some private MC samples with a single file
#for comp in selectedComponents:
#    comp.splitFactor = 1
#    comp.fineSplitFactor = 40

#-------- HOW TO RUN ----------- 
from PhysicsTools.HeppyCore.framework.heppy import getHeppyOption
test = getHeppyOption('test')
if test == '1':
    monoJetSkim.metCut = 0
    comp = DYJetsToLL_M50_HT100to200
    comp.files = comp.files[:1]
    comp.splitFactor = 1
    comp.fineSplitFactor = 1
    selectedComponents = [ comp ]
elif test == '2':
    for comp in selectedComponents:
        comp.files = comp.files[:1]
        comp.splitFactor = 1
        comp.fineSplitFactor = 1
elif test == 'EOS':
    comp = DYJetsToLL_M50#TTJets
    comp.files = comp.files[:1]
    if getHeppyOption('Wigner'):
        print "Will read from WIGNER"
        comp.files = [ 'root://eoscms//eos/cms/store/mc/Phys14DR/DYJetsToLL_M-50_13TeV-madgraph-pythia8/MINIAODSIM/PU20bx25_PHYS14_25_V1-v1/00000/0432E62A-7A6C-E411-87BB-002590DB92A8.root' ]
    else:
        print "Will read from CERN Meyrin"
        comp.files = [ 'root://eoscms//eos/cms/store/mc/Phys14DR/DYJetsToLL_M-50_13TeV-madgraph-pythia8/MINIAODSIM/PU20bx25_PHYS14_25_V1-v1/10000/F675C068-5E6C-E411-B915-0025907DC9AC.root' ]
    os.system("/afs/cern.ch/project/eos/installation/0.3.15/bin/eos.select fileinfo "+comp.files[0].replace("root://eoscms//","/"))
    comp.splitFactor = 1
    comp.fineSplitFactor = 1
    selectedComponents = [ comp ]
elif test == 'SingleMu':
    comp = SingleMu
    comp.files = comp.files[:1]
    comp.splitFactor = 1
    selectedComponents = [ comp ]
elif test == '5':
    for comp in selectedComponents:
        comp.files = comp.files[:5]
        comp.splitFactor = 1
        comp.fineSplitFactor = 5
elif test == '6':
    # test data
    comp = DoubleElectronAB; comp.name = "dataSamplesAll"
    comp.triggers = []
    jetAna.recalibrateJets = False 
    jetAna.smearJets       = False 
    comp.files = [ '/afs/cern.ch/work/e/emanuele/monox/heppy/CMSSW_7_2_3_patch1/src/step5.root' ]
    comp.isMC = False
    comp.splitFactor = 1
    comp.fineSplitFactor = 1
    monoJetSkim.metCut = 0
    selectedComponents = [ comp ]
elif test == 'synch-74X': # sync
    from CMGTools.MonoXAnalysis.samples.samples_monojet_13TeV_74X import *
    #eventSelector.toSelect = [ 11809 ]
    #sequence = cfg.Sequence([eventSelector] + dmCoreSequence+[ ttHEventAna, treeProducer, ])
    monoJetCtrlLepSkim.minLeptons = 0
    monoJetSkim.metCut = 0  
    what = getHeppyOption("sample")
    if what == "ADD":
        comp = ADD_MJ
        comp.files = [ 'root://eoscms//eos/cms/store/relval/CMSSW_7_4_1/RelValADDMonoJet_d3MD3_13/MINIAODSIM/MCRUN2_74_V9_gensim71X-v1/00000/80CF5456-B9EC-E411-93DA-002618FDA248.root' ]
        selectedComponents = [ comp ]
    elif what == "TTLep":
        comp = TTLep
        comp.files = [ 'root://eoscms//eos/cms/store/relval/CMSSW_7_4_1/RelValProdTTbar_13/MINIAODSIM/MCRUN2_74_V9_gensim71X-v1/00000/0A9E2CED-C9EC-E411-A8E4-003048FFCBA8.root' ]
        selectedComponents = [ comp ]
    elif what == "DYJets":
        comp = DYJetsToLL_M50_50ns
        comp.files = [ 'root://eoscms//eos/cms/store/mc/RunIISpring15DR74/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8/MINIAODSIM/Asympt50ns_MCRUN2_74_V9A-v2/60000/04963444-D107-E511-B245-02163E00F339.root' ]
        jetAna.mcGT = "MCRUN2_74_V9A"
        selectedComponents = [ comp ]
    elif what == "TTbar":
        comp = TTbar
        comp.files = [ 'root://eoscms//eos/cms/store/relval/CMSSW_7_4_1/RelValProdTTbar_13/MINIAODSIM/MCRUN2_74_V9_gensim71X-v1/00000/0A9E2CED-C9EC-E411-A8E4-003048FFCBA8.root' ]
        selectedComponents = [ comp ]
    elif what == "WJets":
        comp = WJetsToLNu_HT400to600
        comp.files = [ 'root://eoscms//eos/cms/store/mc/RunIISpring15DR74/WJetsToLNu_HT-400To600_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/MINIAODSIM/Asympt25ns_MCRUN2_74_V9-v3/00000/6408230F-9F08-E511-A1A6-D4AE526A023A.root' ]
        selectedComponents = [ comp ]
    elif what == "RSGrav":
        comp = RSGravGaGa
        comp.files = [ 'root://eoscms//eos/cms/store/relval/CMSSW_7_4_1/RelValRSGravitonToGaGa_13TeV/MINIAODSIM/MCRUN2_74_V9_gensim71X-v1/00000/189277BA-DCEC-E411-B3B8-0025905B859E.root' ]
        selectedComponents = [ comp ]
    else:
        selectedComponents = RelVals741
    jetAna.recalibrateJets = True
    jetAna.smearJets       = False
    for comp in selectedComponents:
        comp.splitFactor = 1
        comp.fineSplitFactor = 10
elif test == 'SR':
    selectedComponents = backgroundSamples + signalSamples
    #selectedComponents = backgroundSamples
    monoJetSkim.metCut = 200
    monoJetCtrlLepSkim.minLeptons = 0
    for comp in selectedComponents:
        comp.splitFactor = 350
elif test == '74X-MC':
    from CMGTools.MonoXAnalysis.samples.samples_monojet_13TeV_74X import *
    what = getHeppyOption("sample")
    if what == "TT":
        monoJetCtrlLepSkim.minLeptons = 0
        selectedComponents = [ TT_bx25 ]
    elif what == "Z":
        monoJetCtrlLepSkim.minLeptons = 0
        monoJetSkim.metCut = 0
        #selectedComponents = [ ZEE_bx25, ZMM_bx25, ZTT_bx25 ]
        selectedComponents = [ ZEE_bx25 ]
    else:
        selectedComponents = RelVals740
    if not getHeppyOption("all"):
        for comp in selectedComponents:
            comp.files = comp.files[:1]
            comp.splitFactor = 1
            comp.fineSplitFactor = 1 if getHeppyOption("single") else 4
elif test == '74X-Data':
    from CMGTools.MonoXAnalysis.samples.samples_monojet_13TeV_74X import *
    from CMGTools.MonoXAnalysis.samples.samples_8TeVReReco_74X import *
    what = getHeppyOption("sample")
    if what == "JetHT":
        monoJetSkim.metCut = 0
        selectedComponents = [ JetHT_742 ]
    elif what == "Z":
        monoJetCtrlLepSkim.minLeptons = 2
        monoJetSkim.metCut = 0
        selectedComponents = [ SingleMuZ_742, DoubleElectronZ_742 ]
    elif what == "MuEG":
        selectedComponents = [ MuEG_742 ]
    elif what == "EGamma":
        selectedComponents = [ privEGamma2015A ]
        lepAna.loose_electron_id = ""
        lepAna.loose_electron_relIso = 1000.
        photonAna.gammaID = "POG_PHYS14_25ns_Loose_NoIso"
        monoJetCtrlLepSkim.minLeptons = 0
        monoJetSkim.metCut = 0
    else:
        selectedComponents = dataSamples742
    for comp in selectedComponents:
        if not getHeppyOption("all"):
            comp.files = comp.files[:1]
            comp.splitFactor = 1
            comp.fineSplitFactor = 1 if getHeppyOption("single") else 8

## output histogram
outputService=[]
from PhysicsTools.HeppyCore.framework.services.tfile import TFileService
output_service = cfg.Service(
    TFileService,
    'outputfile',
    name="outputfile",
    fname='treeProducerDarkMatterMonoJet/tree.root',
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
                     events_class = event_class)



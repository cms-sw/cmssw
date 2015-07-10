import PhysicsTools.HeppyCore.framework.config as cfg


#Load all analyzers
from CMGTools.TTHAnalysis.analyzers.susyCore_modules_cff import *

# Comment this line if you want the diagnostic folders produced along with the output root file
cfg.Analyzer.nosubdir = True


##------------------------------------------
## Redefine what I need
##------------------------------------------

#Vertex
vertexAna.keepFailingEvents = True # keep events with no good vertices

#Lepton
lepAna.loose_muon_dxy = 0.5
lepAna.loose_muon_dz  = 1.0
lepAna.loose_muon_relIso  = 0.15
lepAna.loose_muon_isoCut = lambda muon :muon.miniRelIso < 0.2

lepAna.loose_electron_pt  = 5
lepAna.loose_electron_eta    = 2.4
lepAna.loose_electron_relIso = 0.15
lepAna.loose_electron_isoCut = lambda electron : electron.miniRelIso < 0.1

lepAna.loose_electron_id  = "POG_Cuts_ID_PHYS14_25ns_v1_ConvVetoDxyDz_Veto_full5x5"
lepAna.loose_electron_lostHits = 999. # no cut
lepAna.loose_electron_dxy    = 999.
lepAna.loose_electron_dz     = 999.

lepAna.inclusive_electron_id  = "POG_Cuts_ID_PHYS14_25ns_v1_ConvVetoDxyDz_Veto_full5x5"
lepAna.inclusive_electron_lostHits = 999. # no cut since embedded in ID
lepAna.inclusive_electron_dxy    = 999. # no cut since embedded in ID
lepAna.inclusive_electron_dz     = 999. # no cut since embedded in ID

lepAna.mu_isoCorr = "deltaBeta"
lepAna.ele_isoCorr = "deltaBeta"
lepAna.ele_tightId = "Cuts_PHYS14_25ns_v1_ConvVetoDxyDz"
lepAna.notCleaningElectrons = True
lepAna.doMiniIsolation = True
lepAna.miniIsolationPUCorr = 'raw'

# JET (for event variables do apply the jetID and not PUID yet)
jetAna.relaxJetId = False
jetAna.doPuId = False
jetAna.doQG = True
jetAna.jetEta = 4.7
jetAna.jetEtaCentral = 2.5
jetAna.jetPt = 10.
#jetAna.mcGT = "PHYS14_V4_MC" # jec corrections
jetAna.recalibrateJets = False
jetAna.jetLepDR = 0.4
jetAna.smearJets = False
jetAna.jetGammaDR = 0.4
jetAna.minGammaPt = 20.
jetAna.gammaEtaCentral = 2.4
jetAna.cleanJetsFromFirstPhoton = True
jetAna.cleanJetsFromIsoTracks = True ## added for Dominick

# TAU 
tauAna.inclusive_ptMin = 20.0
tauAna.inclusive_etaMax = 2.3
tauAna.inclusive_dxyMax = 99999.
tauAna.inclusive_dzMax = 99999.
tauAna.inclusive_vetoLeptons = False
tauAna.inclusive_vetoLeptonsPOG = True
tauAna.inclusive_decayModeID = "byLooseCombinedIsolationDeltaBetaCorr3Hits" # ignored if not set or ""
tauAna.inclusive_tauAntiMuonID = "againstMuonTight"
tauAna.inclusive_tauAntiElectronID = "againstElectronLoose"

tauAna.loose_ptMin = 20.0
tauAna.loose_etaMax = 2.3
tauAna.loose_dxyMax = 99999.
tauAna.loose_dzMax = 99999.
tauAna.loose_vetoLeptons = False
tauAna.loose_vetoLeptonsPOG = True
tauAna.loose_decayModeID = "byLooseCombinedIsolationDeltaBetaCorr3Hits" # ignored if not set or ""
tauAna.loose_tauAntiMuonID = "againstMuonTight"
tauAna.loose_tauAntiElectronID = "againstElectronLoose"


# Photon
photonAna.etaCentral = 2.5
photonAna.ptMin = 20
photonAna.gammaID = "POG_PHYS14_25ns_Loose_old_looseSieie_NoIso"
photonAna.do_randomCone = True
photonAna.do_mc_match = True

# Isolated Track
isoTrackAna.setOff=False

# recalibrate MET
metAna.recalibrate = False

# store all taus by default
genAna.allGenTaus = True

# switch off the SV and MC matching
#ttHSVAna.do_mc_match = False

##------------------------------------------ 
##  CONTROL VARIABLES
##------------------------------------------ 

from CMGTools.TTHAnalysis.analyzers.ttHMT2Control import ttHMT2Control

ttHMT2Control = cfg.Analyzer(
            ttHMT2Control, name = 'ttHMT2Control'
            )

##------------------------------------------
##  TOLOLOGIAL VARIABLES: minMT, MT2
##------------------------------------------

from CMGTools.TTHAnalysis.analyzers.ttHTopoVarAnalyzer import ttHTopoVarAnalyzer

ttHTopoJetAna = cfg.Analyzer(
            ttHTopoVarAnalyzer, name = 'ttHTopoVarAnalyzer',
            doOnlyDefault = True
            )

from PhysicsTools.Heppy.analyzers.eventtopology.MT2Analyzer import MT2Analyzer

MT2Ana = cfg.Analyzer(
    MT2Analyzer, name = 'MT2Analyzer',
    doOnlyDefault = True
    )

##------------------------------------------
##  Z skim
##------------------------------------------

from CMGTools.TTHAnalysis.analyzers.ttHmllSkimmer import ttHmllSkimmer
# Tree Producer                                                                                                                                                                         
ttHZskim = cfg.Analyzer(
            ttHmllSkimmer, name='ttHmllSkimmer',
            lepId=[13],
            maxLeps=3,
            massMin=60,
            massMax=120,
            doZGen = False,
            doZReco = True
            )


##------------------------------------------
##  PRODUCER
##------------------------------------------

from CMGTools.RootTools.samples.triggers_13TeV_PHYS14 import triggers_HT900, triggers_MET170, triggers_HTMET, triggers_MT2_mumu, triggers_MT2_ee, triggers_MT2_mue, triggers_1mu, triggers_photon155,triggers_1mu_isolow


triggerFlagsAna.triggerBits = {
            'HT900' : triggers_HT900,
            'MET170' : triggers_MET170,
            'ht350met120' : triggers_HTMET,
            'SingleMu' : triggers_1mu_isolow,
            'DoubleMu' : triggers_MT2_mumu,
            'DoubleEl' : triggers_MT2_ee,
            'MuEG'     : triggers_MT2_mue,
            'htXprescale' : triggers_HTMET,
            'Photons'  : triggers_photon155
}

#-------- SEQUENCE

from CMGTools.TTHAnalysis.analyzers.treeProducerSusyFullHad import *

treeProducer = cfg.Analyzer(
     AutoFillTreeProducer, name='treeProducerSusyFullHad',
##     AutoFillTreeProducer, name='treeProducerSusyCore',
     vectorTree = True,
     saveTLorentzVectors = False,  # can set to True to get also the TLorentzVectors, but trees will be bigger
     PDFWeights = PDFWeights,
     globalVariables = susyFullHad_globalVariables,
     globalObjects = susyFullHad_globalObjects,
     collections = susyFullHad_collections,
     defaultFloatType = 'F',
     treename = 'mt2'
)

susyCoreSequence.insert(susyCoreSequence.index(skimAnalyzer),
                        susyCounter)

#susyCoreSequence.insert(susyCoreSequence.index(ttHLepSkim),
#                        ttHZskim)

#susyCoreSequence.insert(susyCoreSequence.index(ttHCoreEventAna),
#                        ttHSVAna)

sequence = cfg.Sequence(
    susyCoreSequence+[
    ttHMT2Control,
    MT2Ana,
    ttHTopoJetAna,
    ttHFatJetAna,
    treeProducer,
    ])

###---- to switch off the compression
#treeProducer.isCompressed = 0





from PhysicsTools.HeppyCore.framework.heppy import getHeppyOption

#-------- HOW TO RUN
# choose 2 for full production
test = 0
isData = False
doSpecialSettingsForMECCA = 1
if test==0:
    # ------------------------------------------------------------------------------------------- #
    # --- all this lines taken from CMGTools.RootTools.samples.samples_13TeV_PHYS14
    # --- They may not be in synch anymore 
    from CMGTools.RootTools.samples.ComponentCreator import ComponentCreator
    kreator = ComponentCreator()
    testComponent = kreator.makeMCComponent("testComponent", "/GJets_HT-200to400_Tune4C_13TeV-madgraph-tauola/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM", "CMS", ".*root",489.9)
    mcSamples=[testComponent]

    dataDir = os.environ['CMSSW_BASE']+"/src/CMGTools/TTHAnalysis/data"
    json=dataDir+'/json/Cert_Run2012ABCD_22Jan2013ReReco.json'
    from CMGTools.TTHAnalysis.setup.Efficiencies import *

    for comp in mcSamples:
        comp.isMC = True
        comp.isData = False
        comp.splitFactor = 250 
        comp.puFileMC=dataDir+"/puProfile_Summer12_53X.root"
        comp.puFileData=dataDir+"/puProfile_Data12.root"
        comp.efficiency = eff2012
        comp.json = json
    # ------------------------------------------------------------------------------------------- #

    #eventSelector.toSelect = [ 442430994 ]
    #sequence = cfg.Sequence([eventSelector] + sequence)
    comp=testComponent
    #comp.files = ['/afs/cern.ch/user/d/dalfonso/public/TESTfilesPHY14/gjets_ht200to400_miniaodsim_fix.root']
    #comp.files = ['/afs/cern.ch/user/d/dalfonso/public/TESTspring/ttbar25nsmad_1ECE44F9-5F02-E511-9A65-02163E00EA1F.root']
    comp.files = ['/afs/cern.ch/user/d/dalfonso/public/74samples/QCD_Pt_1000to1400_Asympt25ns/0AF8E723-53F9-E411-86B4-0025905C2CBA.root']

    selectedComponents = [comp]
    comp.splitFactor = 10
#    comp.fineSplitFactor = 100

elif test==1:
    # Uncomment the two following lines to run on a specific event
    #eventSelector.toSelect = [ 84142401 ]
    #sequence = cfg.Sequence([eventSelector] + sequence)
    #from CMGTools.TTHAnalysis.samples.samples_13TeV_PHYS14 import *
    from CMGTools.RootTools.samples.samples_13TeV_74X import *
    from CMGTools.RootTools.samples.samples_8TeVReReco_74X import *

#    comp=GJets_HT200to400
#    comp.files = ['/afs/cern.ch/user/d/dalfonso/public/TESTfilesPHY14/gjets_ht200to400_miniaodsim_fix.root']

    comp=TTJets
    #comp.files = ['/afs/cern.ch/user/d/dalfonso/public/TESTfilesPHY14/TTJets_miniAOD_fixPhoton_forSynch.root']
    comp.files = ['/afs/cern.ch/user/d/dalfonso/public/TESTspring/ttbar25nsmad_1ECE44F9-5F02-E511-9A65-02163E00EA1F.root']
#    comp.files = ['/afs/cern.ch/user/d/dalfonso/public/74samples/JetHT_GR_R_74_V12_19May_RelVal/1294BDDB-B7FE-E411-8028-002590596490.root']

    selectedComponents = [comp]
    comp.splitFactor = 1
#    comp.triggers = triggers_HT900 + triggers_HTMET + triggers_photon155 + triggers_1mu_isolow + triggers_MT2_mumu + triggers_MT2_ee + triggers_MT2_mue # to apply trigger skimming

elif test==2:

    #from CMGTools.TTHAnalysis.samples.samples_13TeV_PHYS14 import *
    # full production
#    selectedComponents = [ 
#TTJets, # TTJets
#TToLeptons_tch, TToLeptons_sch, TBarToLeptons_tch, TBarToLeptons_sch, TBar_tWch, T_tWch, #singleTop
#TTWJets, TTZJets, TTH, #TT+boson
#ZJetsToNuNu_HT100to200, ZJetsToNuNu_HT200to400, ZJetsToNuNu_HT400to600, ZJetsToNuNu_HT600toInf, # ZJetsToNuNu_HT
#WJetsToLNu_HT100to200, WJetsToLNu_HT200to400, WJetsToLNu_HT400to600, WJetsToLNu_HT600toInf, # WJetsToLNu_HT
#GJets_HT100to200_fixPhoton, GJets_HT200to400_fixPhoton, GJets_HT400to600_fixPhoton, GJets_HT600toInf_fixPhoton, # GJets_HT
#QCD_HT_100To250_fixPhoton, QCD_HT_250To500_fixPhoton, QCD_HT_500To1000_fixPhoton, QCD_HT_1000ToInf_fixPhoton, QCD_HT_250To500_ext1_fixPhoton, QCD_HT_500To1000_ext1_fixPhoton,QCD_HT_1000ToInf_ext1_fixPhoton, # QCD_HT
#QCD_Pt170to300_fixPhoton, QCD_Pt300to470_fixPhoton, QCD_Pt470to600_fixPhoton, QCD_Pt600to800_fixPhoton, QCD_Pt800to1000_fixPhoton, QCD_Pt1000to1400_fixPhoton, QCD_Pt1400to1800_fixPhoton, QCD_Pt1800to2400_fixPhoton, QCD_Pt2400to3200_fixPhoton, QCD_Pt3200_fixPhoton, # QCD_Pt
#QCD_Pt50to80, QCD_Pt80to120, QCD_Pt120to170, #For QCD Estimate
#SMS_T2tt_2J_mStop850_mLSP100, SMS_T2tt_2J_mStop650_mLSP325, SMS_T2tt_2J_mStop500_mLSP325, SMS_T2tt_2J_mStop425_mLSP325, SMS_T2qq_2J_mStop600_mLSP550, SMS_T2qq_2J_mStop1200_mLSP100, SMS_T2bb_2J_mStop900_mLSP100, SMS_T2bb_2J_mStop600_mLSP580, SMS_T1tttt_2J_mGl1500_mLSP100, SMS_T1tttt_2J_mGl1200_mLSP800, SMS_T1qqqq_2J_mGl1400_mLSP100, SMS_T1qqqq_2J_mGl1000_mLSP800, SMS_T1bbbb_2J_mGl1500_mLSP100, SMS_T1bbbb_2J_mGl1000_mLSP900, # SMS
#DYJetsToLL_M50_HT100to200, DYJetsToLL_M50_HT200to400, DYJetsToLL_M50_HT400to600, DYJetsToLL_M50_HT600toInf # DYJetsToLL_M50_HT
#]

    from CMGTools.RootTools.samples.samples_13TeV_74X import *
    selectedComponents = [ 
TTJets, TTJets_LO, # TTJets
QCD_Pt80to120, QCD_Pt120to170, QCD_Pt300to470, QCD_Pt470to600, QCD_Pt1000to1400, QCD_Pt1400to1800, QCD_Pt1800to2400, QCD_Pt2400to3200, QCD_Pt3200toInf, # QCD_Pt
]

    # test all components (1 thread per component).
    for comp in selectedComponents:
        comp.splitFactor = 1200
        #comp.fineSplitFactor = 2 # to run two jobs per file
        comp.files = comp.files[:]
        #comp.files = comp.files[:1]
        #comp.files = comp.files[57:58]  # to process only file [57]  
        # triggers on MC
        #comp.triggers = triggers_HT900 + triggers_HTMET + triggers_photon155 + triggers_1mu_isolow + triggers_MT2_mumu + triggers_MT2_ee + triggers_MT2_mue # to apply trigger skimming

elif test==3:
    # run on data
    isData = True
    from CMGTools.TTHAnalysis.samples.samples_13TeV_74X import *
    selectedComponents = [ jetHT_0T ]
  



# ------------------------------------------------------------------------------------------- #


if doSpecialSettingsForMECCA:
    jetAna.doQG = False
    photonAna.do_randomCone = False



if isData:
    jetAna.recalibrateJets = False
    photonAna.do_mc_match = False
    for comp in mcSamples:
        comp.isMC = False
        comp.isData = True
        #comp.files = ['/afs/cern.ch/user/d/dalfonso/public/74samples/JetHT_GR_R_74_V12_19May_RelVal/1294BDDB-B7FE-E411-8028-002590596490.root']
# ------------------------------------------------------------------------------------------- #








from PhysicsTools.HeppyCore.framework.services.tfile import TFileService 
output_service = cfg.Service(
      TFileService,
      'outputfile',
      name="outputfile",
      fname='mt2.root',
      option='recreate'
    )

# the following is declared in case this cfg is used in input to the heppy.py script
from PhysicsTools.HeppyCore.framework.eventsfwlite import Events
from CMGTools.TTHAnalysis.tools.EOSEventsWithDownload import EOSEventsWithDownload
event_class = EOSEventsWithDownload
if getHeppyOption("nofetch"):
    event_class = Events
config = cfg.Config( components = selectedComponents,
                     sequence = sequence,
                     services = [output_service],
                     events_class = event_class)
#                     events_class = Events)
#printComps(config.components, True)

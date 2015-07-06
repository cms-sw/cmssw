import PhysicsTools.HeppyCore.framework.config as cfg

#Load all analyzers
from CMGTools.ObjectStudies.analyzers.metCoreModules_cff import *

cfg.Analyzer.nosubdir = True

##------------------------------------------
##  PRODUCER
##------------------------------------------

from CMGTools.TTHAnalysis.samples.samples_13TeV_PHYS14 import triggers_1mu, triggers_mumu_iso, triggers_1mu_isolow

triggers_8TeV_mumu = ["HLT_Mu17_Mu8_v*","HLT_Mu17_TkMu8_v*"]
triggers_8TeV_1mu = [ 'HLT_IsoMu24_eta2p1_v*' ]

triggerFlagsAna.triggerBits = {
            'SingleMu' : triggers_8TeV_1mu,
            'DoubleMu' : triggers_8TeV_mumu,
}

#-------- SEQUENCE

from CMGTools.ObjectStudies.analyzers.treeProducerMET import *

treeProducer = cfg.Analyzer(
     AutoFillTreeProducer, name='treeProducerMET',
     vectorTree = True,
     saveTLorentzVectors = False,  # can set to True to get also the TLorentzVectors, but trees will be bigger
     PDFWeights = PDFWeights,
     globalVariables = met_globalVariables,
     globalObjects = met_globalObjects,
     collections = met_collections,
     defaultFloatType = 'F',
)

## same as MET perf analyszer
treeProducer.treename = 'Events'

#-------- SEQUENCE

metAna.doTkMet = True
metAna.doSpecialMet = False

metSequence = cfg.Sequence(
    metCoreSequence +[treeProducer]
    )

# -------------------- lepton modules below needed for the DoubleMuon

ttHLepSkim.minLeptons = 2

metSequence.insert(metSequence.index(lepAna)+1,
                   ttHLepSkim)
metSequence.insert(metSequence.index(lepAna)+2,
                   ttHZskim)

# -------------------- lepton modules below needed for the WJets

#ttHLepSkim.minLeptons = 1

#metSequence.insert(metSequence.index(lepAna)+1,
#                   ttHLepSkim)

###---- to switch off the comptrssion
#treeProducer.isCompressed = 0

#-------- SAMPLES AND TRIGGERS -----------
from CMGTools.RootTools.samples.samples_13TeV_74X import *
from CMGTools.RootTools.samples.samples_8TeVReReco_74X import * # <-- this one for the official sample
from CMGTools.ObjectStudies.samples.samples_METPOG_private import * #<-- this one for the private re-reco

#-------- HOW TO RUN

test = 3

if test==0:
    selectedComponents = [DoubleMu_742, DoubleMu_740p9]
#    selectedComponents = [ DoubleMuParked_1Apr_RelVal_dm2012D_v2_newPFHCalib , DoubleMuParked_1Apr_RelVal_dm2012D_v2_oldPFHCalib , DoubleMuparked_1Apr_RelVal_dm2012D_v2 ]
    for comp in selectedComponents:
        comp.splitFactor = 251
        comp.files = comp.files[:]
        comp.triggers = triggers_8TeV_mumu

elif test==1:
    selectedComponents = [ RelValZMM_7_4_1,RelValZMM_7_4_0_pre9 ]
#    selectedComponents = [RelVal_741_Philfixes]
#    selectedComponents = relValkate
    for comp in selectedComponents:
#        comp.splitFactor = 1
        comp.splitFactor = 100
        comp.files = comp.files[:]


   # ----------------------- Summer15 options -------------------------------------------------------------------- #

elif test==2:
    # test a single component, using a single thread.
    ## 40 ns ttbar DY
#    comp=DYJetsToLL_M50_PU4bx50
#    comp.files = comp.files[:1]
    ## 25 ns ttbar PHYS14
#    comp = TTJets
#    comp.files = comp.files[:1]
    comp=TTJets
    comp.files = ['/afs/cern.ch/work/d/dalfonso/public/ttjets_miniaodsim_00C90EFC-3074-E411-A845-002590DB9262.root']
    selectedComponents = [comp]
    comp.splitFactor = 1

elif test==3:
    # test all components (1 thread per component).
    selectedComponents = [ DYJetsToLL_M50_50ns ]
    for comp in selectedComponents:
#        comp.splitFactor = 1
#        comp.files = comp.files[:1]
        comp.splitFactor = 1000
        comp.files = comp.files[:]



elif test==4:
    # test all components (1 thread per component).
    # ---> this is for the sample that do not need the Zskim
    selectedComponents = [ WJetsToLNu_50ns ]
    for comp in selectedComponents:
        comp.splitFactor = 1000
        comp.files = comp.files[:]
#        comp.splitFactor = 1
#        comp.files = comp.files[:1]

    # ------------------------------------------------------------------------------------------- #


from PhysicsTools.HeppyCore.framework.services.tfile import TFileService 
output_service = cfg.Service(
      TFileService,
      'outputfile',
      name="outputfile",
      fname='METtree.root',
      option='recreate'
    )


# the following is declared in case this cfg is used in input to the heppy.py script                                                                                           
from PhysicsTools.HeppyCore.framework.eventsfwlite import Events

# -------------------- Running Download from EOS

from PhysicsTools.HeppyCore.framework.heppy import getHeppyOption
from CMGTools.TTHAnalysis.tools.EOSEventsWithDownload import EOSEventsWithDownload
event_class = EOSEventsWithDownload
if getHeppyOption("nofetch"):
    event_class = Events 


# -------------------- Running pre-processor

#from PhysicsTools.Heppy.utils.cmsswPreprocessor import CmsswPreprocessor
#preprocessor = CmsswPreprocessor("$CMSSW_BASE/src/CMGTools/ObjectStudies/cfg/MetType1_dump.py")

config = cfg.Config( components = selectedComponents,
                     sequence = metSequence,
                     services = [output_service],
#                     preprocessor=preprocessor, # comment if pre-processor non needed
#                     events_class = event_class)
                     events_class = Events)

#printComps(config.components, True)
        

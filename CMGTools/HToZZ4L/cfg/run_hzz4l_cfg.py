##########################################################
##       CONFIGURATION FOR HZZ4L TREES                  ##
##########################################################
import PhysicsTools.HeppyCore.framework.config as cfg

#Load all analyzers
from CMGTools.HToZZ4L.analyzers.hzz4lCore_modules_cff import * 


#-------- SAMPLES AND TRIGGERS -----------

#-------- SEQUENCE
from CMGTools.HToZZ4L.samples.samples_13TeV_Spring15 import *

selectedComponents = mcSamples + dataSamples
sequence = cfg.Sequence(hzz4lCoreSequence)

for comp in mcSamples:
    comp.triggers = []
    comp.vetoTriggers = []

from PhysicsTools.HeppyCore.framework.heppy_loop import getHeppyOption
test = getHeppyOption('test')
if test == "1":
    comp = QQHZZ4L
    comp.files = comp.files[:1]
    comp.splitFactor = 1
    comp.fineSplitFactor = 1 if getHeppyOption('single') else 5
    selectedComponents = [ comp ]
    if getHeppyOption('events'):
        eventSelector.toSelect = [ eval("("+x.replace(":",",")+")") for x in getHeppyOption('events').split(",") ]
        sequence = cfg.Sequence([eventSelector] + hzz4lCoreSequence)
        print "Will select events ",eventSelector.toSelect
elif test == '2':
    for comp in selectedComponents:
        comp.files = comp.files[:1]
        comp.splitFactor = 1
        comp.fineSplitFactor = 1
elif test=="sync":
    comp = GGHZZ4L
    comp.name = 'HZZ4L'
    comp.files = [ 'root://eoscms.cern.ch//eos/cms'+X for X in (
         #'/store/mc/RunIISpring15DR74/GluGluHToZZTo4L_M125_13TeV_powheg_JHUgen_pythia8/MINIAODSIM/Asympt25ns_MCRUN2_74_V9-v1/10000/00E6807C-9C16-E511-A165-0025905938A4.root',
         #'/store/cmst3/group/hzz/GluGluHToZZTo4L_M125_13TeV_powheg_JHUgen_pythia8_RunIISpring15DR74_Asympt25ns_MCRUN2_74_V9-v1_MINIAODSIM_00E6807C-9C16-E511-A165-0025905938A4.root',
         '/store/mc/RunIISpring15DR74/VBF_HToZZTo4L_M125_13TeV_powheg_JHUgen_pythia8/MINIAODSIM/Asympt25ns_MCRUN2_74_V9-v1/00000/68791C0A-3013-E511-88FD-D4AE5269F5FF.root', 
         '/store/mc/RunIISpring15DR74/WplusH_HToZZTo4L_M125_13TeV_powheg-minlo-HWJ_JHUgen_pythia8/MINIAODSIM/Asympt25ns_MCRUN2_74_V9-v1/60000/04BD6860-9F08-E511-8A80-842B2B1858FB.root',
         '/store/mc/RunIISpring15DR74/WminusH_HToZZTo4L_M125_13TeV_powheg-minlo-HWJ_JHUgen_pythia8/MINIAODSIM/Asympt25ns_MCRUN2_74_V9-v2/70000/4A9FED55-DF0C-E511-A4B2-3417EBE6471D.root',
         '/store/mc/RunIISpring15DR74/ZH_HToZZ_4LFilter_M125_13TeV_powheg-minlo-HZJ_JHUgen_pythia8/MINIAODSIM/Asympt25ns_MCRUN2_74_V9-v1/00000/104B7067-0C02-E511-8FFB-0030487D07BA.root',
    )]
    #comp.splitFactor = 9
    #comp.fineSplitFactor = 1 if getHeppyOption('single') else 5
    comp.splitFactor = 1 if getHeppyOption('single') else 5
    selectedComponents = [ comp ]
    if getHeppyOption('events'):
        eventSelector.toSelect = [ eval("("+x.replace(":",",")+")") for x in getHeppyOption('events').split(",") ]
        sequence = cfg.Sequence([eventSelector] + hzz4lCoreSequence)
        print "Will select events ",eventSelector.toSelect

# the following is declared in case this cfg is used in input to the heppy.py script
from PhysicsTools.HeppyCore.framework.eventsfwlite import Events
config = cfg.Config( components = selectedComponents,
                     sequence = sequence,
                     services = [],  
                     events_class = Events)



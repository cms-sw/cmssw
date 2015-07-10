##########################################################
##       GENERIC SUSY TREE PRODUCTION CONFIG.           ##
## no skim is applied in this configuration, as it is   ##
## meant only to check that all common modules run ok   ##
##########################################################


import CMGTools.RootTools.fwlite.Config as cfg
from CMGTools.RootTools.fwlite.Config import printComps
from CMGTools.RootTools.RootTools import *
from PhysicsTools.HeppyCore.framework.heppy import getHeppyOption

#Load all common analyzers
from CMGTools.VVResonances.analyzers.core_cff import * 

#-------- SAMPLES AND TRIGGERS -----------
from CMGTools.VVResonances.samples.samples_13TeV_Spring15 import * 

selectedComponents = mcSamples

#-------- Analyzer
from CMGTools.VVResonances.analyzers.tree_cff import * 

#-------- SEQUENCE

sequence = cfg.Sequence(coreSequence+[vvSkimmer,vvTreeProducer])


#-------- HOW TO RUN
test = 0
if test==1:
    # test a single component, using a single thread.
    comp = RSGravToWWToLNQQ_2000
    comp.files = comp.files[:1]
    selectedComponents = [comp]
    comp.splitFactor = 1

elif test==2:    
    # test all components (1 thread per component).
    for comp in selectedComponents:
        comp.splitFactor = 1
        comp.files = comp.files[:1]





## output histogram
outputService=[]
from PhysicsTools.HeppyCore.framework.services.tfile import TFileService
output_service = cfg.Service(
    TFileService,
    'outputfile',
    name="outputfile",
    fname='vvTreeProducer/tree.root',
    option='recreate'
    )    
outputService.append(output_service)



from PhysicsTools.HeppyCore.framework.eventsfwlite import Events
from CMGTools.TTHAnalysis.tools.EOSEventsWithDownload import EOSEventsWithDownload
event_class = EOSEventsWithDownload
if getHeppyOption("nofetch"):
    event_class = Events 
config = cfg.Config( components = selectedComponents,
                     sequence = sequence,
                     services = [],  
                     events_class = event_class)


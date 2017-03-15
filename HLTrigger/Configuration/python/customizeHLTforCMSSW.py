import FWCore.ParameterSet.Config as cms

# helper fuctions
from HLTrigger.Configuration.common import *

# add one customisation function per PR
# - put the PR number into the name of the function
# - add a short comment

# Matching ECAL selective readout in particle flow, need a new input with online Selective Readout Flags
def customiseFor17794(process):
     for edproducer in process._Process__producers.values():
         if hasattr(edproducer,'producers'):
             for pset in edproducer.producers:
                 if (pset.name == 'PFEBRecHitCreator' or pset.name == 'PFEERecHitCreator'):
                     if not hasattr(pset,'srFlags'):
                         pset.srFlags = cms.InputTag('hltEcalDigis')
     return process


# CMSSW version specific customizations
def customizeHLTforCMSSW(process, menuType="GRun"):
    # add call to action function in proper order: newest last!
    process = customiseFor17794(process)

    return process

import FWCore.ParameterSet.Config as cms

# helper fuctions
from HLTrigger.Configuration.common import *

# add one customisation function per PR
# - put the PR number into the name of the function
# - add a short comment
# for example:

# CCCTF tuning
# def customiseFor12718(process):
#     for pset in process._Process__psets.values():
#         if hasattr(pset,'ComponentType'):
#             if (pset.ComponentType == 'CkfBaseTrajectoryFilter'):
#                 if not hasattr(pset,'minGoodStripCharge'):
#                     pset.minGoodStripCharge = cms.PSet(refToPSet_ = cms.string('HLTSiStripClusterChargeCutNone'))
#     return process

# Matching ECAL selective readout in particle flow, need a new input with online Selective Readout Flags
def customiseFor18016(process):
     for edproducer in producers_by_type(process, "PFRecHitProducer"):
          if hasattr(edproducer,'producers'):
               for pset in edproducer.producers:
                    if (pset.name == 'PFEBRecHitCreator' or pset.name == 'PFEERecHitCreator'):
                         if not hasattr(pset,'srFlags'):
                              pset.srFlags = cms.InputTag('hltEcalDigis')
     return process


# CMSSW version specific customizations
def customizeHLTforCMSSW(process, menuType="GRun"):
    # add call to action function in proper order: newest last!
    # process = customiseFor12718(process)
    process = customiseFor18016(process)

    return process

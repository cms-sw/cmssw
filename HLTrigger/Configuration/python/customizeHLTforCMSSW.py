import FWCore.ParameterSet.Config as cms

# helper functions
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



def customiseForOffline(process):
    # For running HLT offline on Run-3 Data, use "(OnlineBeamSpotESProducer).timeThreshold = 1e6",
    # in order to pick the beamspot that was actually used by the HLT (instead of a "fake" beamspot).
    # These same settings can be used offline for Run-3 Data and Run-3 MC alike.
    # Note: the products of the OnlineBeamSpotESProducer are used only
    #       if the configuration uses "(BeamSpotOnlineProducer).useTransientRecord = True".
    # See CMSHLT-2271 and CMSHLT-2300 for further details.
    for prod in esproducers_by_type(process, 'OnlineBeamSpotESProducer'):
        prod.timeThreshold = int(1e6)

    # For running HLT offline and relieve the strain on Frontier so it will no longer inject a
    # transaction id which tells Frontier to add a unique "&freshkey" to many query URLs.
    # That was intended as a feature to only be used by the Online HLT, to guarantee that fresh conditions
    # from the database were loaded at each Lumi section
    # Seee CMSHLT-3123 for further details
    if hasattr(process, 'GlobalTag'):
        # Set ReconnectEachRun and RefreshEachRun to False
        process.GlobalTag.ReconnectEachRun = cms.untracked.bool(False)
        process.GlobalTag.RefreshEachRun = cms.untracked.bool(False)

        if hasattr(process.GlobalTag, 'toGet'):
            # Filter out PSet objects containing only 'record' and 'refreshTime'
            process.GlobalTag.toGet = [
                pset for pset in process.GlobalTag.toGet
                if set(pset.parameterNames_()) != {'record', 'refreshTime'}
            ]

    return process

def customizeHLTfor44576(process):
    """Ensure TrackerAdditionalParametersPerDetRcd ESProducer is run when needed"""
    for esprod in esproducers_by_type(process, 'TrackerGeometricDetESModule'):
        process.load("Geometry.TrackerGeometryBuilder.TrackerAdditionalParametersPerDet_cfi")
        break
    return process

def customizeHLTfor45063(process):
    """Assigns value of MuonHLTSeedMVAClassifier mva input file, scales and mean values according to the value of isFromL1"""
    for prod in producers_by_type(process, 'MuonHLTSeedMVAClassifier'):
        if hasattr(prod, "isFromL1"):
            if (prod.isFromL1 == True):
                if hasattr(prod, "mvaFileBL1"):
                    prod.mvaFileB = prod.mvaFileBL1
                if hasattr(prod, "mvaFileEL1"):
                    prod.mvaFileE = prod.mvaFileEL1
                if hasattr(prod, "mvaScaleMeanBL1"):
                    prod.mvaScaleMeanB = prod.mvaScaleMeanBL1
                if hasattr(prod, "mvaScaleStdBL1"):
                    prod.mvaScaleStdB = prod.mvaScaleStdBL1
                if hasattr(prod, "mvaScaleMeanEL1"):
                    prod.mvaScaleMeanE = prod.mvaScaleMeanEL1
                if hasattr(prod, "mvaScaleStdEL1"):                    
                    prod.mvaScaleStdE = prod.mvaScaleStdEL1                
            else:
                if hasattr(prod, "mvaFileBL2"):
                    prod.mvaFileB = prod.mvaFileBL2
                if hasattr(prod, "mvaFileEL2"):
                    prod.mvaFileE = prod.mvaFileEL2
                if hasattr(prod, "mvaScaleMeanBL2"):
                    prod.mvaScaleMeanB = prod.mvaScaleMeanBL2
                if hasattr(prod, "mvaScaleStdBL2"):
                    prod.mvaScaleStdB = prod.mvaScaleStdBL2
                if hasattr(prod, "mvaScaleMeanEL2"):
                    prod.mvaScaleMeanE = prod.mvaScaleMeanEL2
                if hasattr(prod, "mvaScaleStdEL2"):
                    prod.mvaScaleStdE = prod.mvaScaleStdEL2
                    
    for prod in producers_by_type(process, 'MuonHLTSeedMVAClassifier'):
        delattr(prod,"mvaFileBL1")
        delattr(prod,"mvaFileEL1")
        delattr(prod,"mvaScaleMeanBL1")
        delattr(prod,"mvaScaleStdBL1")
        delattr(prod,"mvaScaleMeanEL1")
        delattr(prod,"mvaScaleStdEL1")
        delattr(prod,"mvaFileBL2")
        delattr(prod,"mvaFileEL2")
        delattr(prod,"mvaScaleMeanBL2")
        delattr(prod,"mvaScaleStdBL2")
        delattr(prod,"mvaScaleMeanEL2")
        delattr(prod,"mvaScaleStdEL2")       
                    
    return process
            

# CMSSW version specific customizations
def customizeHLTforCMSSW(process, menuType="GRun"):

    process = customiseForOffline(process)

    # add call to action function in proper order: newest last!
    # process = customiseFor12718(process)

    process = customizeHLTfor44576(process)
    process = customizeHLTfor45063(process)

    return process

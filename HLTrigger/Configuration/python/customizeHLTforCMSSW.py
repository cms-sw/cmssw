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


def customiseFor2017DtUnpacking(process):
    """Adapt the HLT to run the legacy DT unpacking
    for pre2018 data/MC workflows as the default"""

    if hasattr(process,'hltMuonDTDigis'):
        process.hltMuonDTDigis = cms.EDProducer( "DTUnpackingModule",
            useStandardFEDid = cms.bool( True ),
            maxFEDid = cms.untracked.int32( 779 ),
            inputLabel = cms.InputTag( "rawDataCollector" ),
            minFEDid = cms.untracked.int32( 770 ),
            dataType = cms.string( "DDU" ),
            readOutParameters = cms.PSet(
                localDAQ = cms.untracked.bool( False ),
                debug = cms.untracked.bool( False ),
                rosParameters = cms.PSet(
                    localDAQ = cms.untracked.bool( False ),
                    debug = cms.untracked.bool( False ),
                    writeSC = cms.untracked.bool( True ),
                    readDDUIDfromDDU = cms.untracked.bool( True ),
                    readingDDU = cms.untracked.bool( True ),
                    performDataIntegrityMonitor = cms.untracked.bool( False )
                    ),
                performDataIntegrityMonitor = cms.untracked.bool( False )
                ),
            dqmOnly = cms.bool( False )
        )

    return process

def customiseFor25811(process):
    for prod in producers_by_type(process, "SiPixelClusterProducer"):
        if hasattr(prod, "MissCalibrate") and not prod.MissCalibrate.isTracked():
            prod.MissCalibrate = cms.bool(prod.MissCalibrate.value())
    return process

# default parameters moved from code to configs
def customiseFor24501(process):
    for producer in producers_by_type(process, "RecoTauJetRegionProducer"):
        producer.verbosity = cms.int32(0)
    for producer in producers_by_type(process, "PFRecoTauChargedHadronProducer"):
        producer.verbosity = cms.int32(0)
    for producer in producers_by_type(process, "RecoTauPiZeroProducer"):
        producer.verbosity = cms.int32(0)
        for i in range(0,len(producer.builders)):
            producer.builders[i].verbosity = cms.int32(0)
    for producer in producers_by_type(process, "RecoTauProducer"):
        producer.verbosity = cms.int32(0)
        for i in range(0,len(producer.builders)):
            producer.builders[i].verbosity = cms.int32(0)
    for producer in producers_by_type(process, "PFRecoTauDiscriminationByHPSSelection"):
        producer.verbosity = cms.int32(0)
        for i in range(0,len(producer.decayModes)):
            if not hasattr(producer.decayModes[i],"minPi0Mass"):
                producer.decayModes[i].minPi0Mass = cms.double(-1.e3)
            if not hasattr(producer.decayModes[i],"maxPi0Mass"):
                producer.decayModes[i].maxPi0Mass = cms.double(1.e9)        
            if not hasattr(producer.decayModes[i],"assumeStripMass"):
                producer.decayModes[i].assumeStripMass = cms.double(-1)
    for producer in producers_by_type(process, "RecoTauCleaner"):
        producer.outputSelection = cms.string("")
        producer.verbosity = cms.int32(0)
        for i in range(0,len(producer.cleaners)):
            if not hasattr(producer.cleaners[i],"tolerance"):
                producer.cleaners[i].tolerance = cms.double(0)
    for producer in producers_by_type(process, "PFRecoTauDiscriminationByIsolation"):
         producer.verbosity = cms.int32(0)
         if not hasattr(producer,"storeRawOccupancy"):
             producer.storeRawOccupancy = cms.bool(False)
         if not hasattr(producer,"storeRawSumPt"):
             producer.storeRawSumPt = cms.bool(False)
         if not hasattr(producer,"storeRawPUsumPt"):
             producer.storeRawPUsumPt = cms.bool(False)
         if not hasattr(producer,"storeRawFootprintCorrection"):
             producer.storeRawFootprintCorrection = cms.bool(False)
         if not hasattr(producer,"storeRawPhotonSumPt_outsideSignalCone"):
             producer.storeRawPhotonSumPt_outsideSignalCone = cms.bool(False)
    return process


# CMSSW version specific customizations
def customizeHLTforCMSSW(process, menuType="GRun"):

    # add call to action function in proper order: newest last!
    # process = customiseFor12718(process)
    process = customiseFor24501(process)

    process = customiseFor25811(process)

    return process

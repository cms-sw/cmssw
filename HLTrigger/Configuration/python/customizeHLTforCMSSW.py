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

def customizeHLTfor47611(process):
    """ This customizer
        - adds the CAGeometry ESProducer;
        - cleanup the CANtupletAlpaka producers paramters;
        - add the average sizes paramters to the CANtupletAlpaka producers.
    """
 
    ca_producers_pp = ['CAHitNtupletAlpakaPhase1@alpaka','alpaka_serial_sync::CAHitNtupletAlpakaPhase1']
    ca_producers_hi = ['CAHitNtupletAlpakaHIonPhase1@alpaka','alpaka_serial_sync::CAHitNtupletAlpakaHIonPhase1']
    ca_producers = ca_producers_pp + ca_producers_hi
    ca_parameters = [ 'CAThetaCutBarrel', 'CAThetaCutForward', 
        'dcaCutInnerTriplet', 'dcaCutOuterTriplet', 
        'doPtCut', 'doZ0Cut', 'idealConditions', 
        'includeJumpingForwardDoublets', 'phiCuts','doClusterCut','CPE'] 

    has_pp_producers = False
    has_hi_producers = False

    for ca_producer in ca_producers:
        for prod in producers_by_type(process, ca_producer):

            for par in ca_parameters:
                if hasattr(prod, par):
                    delattr(prod,par)
            
            if not hasattr(prod, 'caGeometry'):
                setattr(prod, 'caGeometry', cms.string('hltCAGeometry'))
            
            for par in ['minYsizeB2','minYsizeB1']:
                if hasattr(prod, par):
                    v = getattr(prod, par)
                    delattr(prod, par)
                    setattr(prod, par, cms.uint32(v.value()))

            if not hasattr(prod, 'dzdrFact'):
                setattr(prod, 'dzdrFact', cms.double(8.0 * 0.0285 / 0.015))
            if not hasattr(prod, 'maxDYsize12'):
                setattr(prod, 'maxDYsize12', cms.uint32(28))
            if not hasattr(prod, 'maxDYsize'):
                setattr(prod, 'maxDYsize', cms.uint32(20))
            if not hasattr(prod, 'maxDYPred'):
                setattr(prod, 'maxDYPred', cms.uint32(20))
            
            if hasattr(prod, 'maxNumberOfDoublets'):
                v = getattr(prod, 'maxNumberOfDoublets')
                delattr(prod, 'maxNumberOfDoublets')
                setattr(prod, 'maxNumberOfDoublets', cms.string(str(v.value())))
            
    for ca_producer in ca_producers_pp:
        for prod in producers_by_type(process, ca_producer):

            has_pp_producers = True

            if not hasattr(prod, 'maxNumberOfTuples'):
                setattr(prod,'maxNumberOfTuples',cms.string(str(32*1024)))
                
            if not hasattr(prod, 'avgCellsPerCell'):
                setattr(prod, 'avgCellsPerCell', cms.double(0.071))
            
            if not hasattr(prod, 'avgCellsPerHit'):
                setattr(prod, 'avgCellsPerHit', cms.double(27))
            
            if not hasattr(prod, 'avgHitsPerTrack'):
                setattr(prod, 'avgHitsPerTrack', cms.double(4.5))
            
            if not hasattr(prod, 'avgTracksPerCell'):
                setattr(prod, 'avgTracksPerCell', cms.double(0.127))

    for ca_producer in ca_producers_hi:
        for prod in producers_by_type(process, ca_producer):

            if has_pp_producers:
                raise Exception("CAHitNtupletAlpaka producers found in the menu for both HIonPhase1 and Phase1. This is not expected!")
            has_hi_producers = True
            
            if not hasattr(prod, 'maxNumberOfTuples'):
                setattr(prod,'maxNumberOfTuples',cms.string(str(256 * 1024))) # way too much, could be ~20k
                
            if not hasattr(prod, 'avgCellsPerCell'):
                setattr(prod, 'avgCellsPerCell', cms.double(0.5))
            
            if not hasattr(prod, 'avgCellsPerHit'):
                setattr(prod, 'avgCellsPerHit', cms.double(40))
            
            if not hasattr(prod, 'avgHitsPerTrack'):
                setattr(prod, 'avgHitsPerTrack', cms.double(5.0))
            
            if not hasattr(prod, 'avgTracksPerCell'):
                setattr(prod, 'avgTracksPerCell', cms.double(0.5))

   
    if has_pp_producers:
        
        process.hltCAGeometry = cms.ESProducer('CAGeometryESProducer@alpaka',
            startingPairs = cms.vint32( [i for i in range(8)] + [13, 14, 15, 16, 17, 18, 19]),
            pairGraph = cms.vint32( 0, 1, 0, 4, 0,
                7, 1, 2, 1, 4,
                1, 7, 4, 5, 7,
                8, 2, 3, 2, 4,
                2, 7, 5, 6, 8,
                9, 0, 2, 1, 3,
                0, 5, 0, 8, 
                4, 6, 7, 9 
            ),
            phiCuts = cms.vint32( 
                522, 730, 730, 522, 626,
                626, 522, 522, 626, 626,
                626, 522, 522, 522, 522,
                522, 522, 522, 522
            ),
            minZ = cms.vdouble(
                    -20., 0., -30., -22., 10., 
                    -30., -70., -70., -22., 15., 
                    -30, -70., -70., -20., -22., 
                    0, -30., -70., -70.
            ),
            maxZ = cms.vdouble( 20., 30., 0., 22., 30., 
                -10., 70., 70., 22., 30., 
                -15., 70., 70., 20., 22., 
                30., 0., 70., 70.),
            maxR = cms.vdouble(20., 9., 9., 20., 7., 
                7., 5., 5., 20., 6., 
                6., 5., 5., 20., 20., 
                9., 9., 9., 9.),
            appendToDataLabel = cms.string('hltCAGeometry'),
            alpaka = cms.untracked.PSet(
                backend = cms.untracked.string(''),
                synchronize = cms.optional.untracked.bool
            )
        )
    
    elif has_hi_producers:

        process.hltCAGeometryESProducer = cms.ESProducer("CAGeometryESProducer@alpaka",
                alpaka = cms.untracked.PSet(
                    backend = cms.untracked.string(''),
                    synchronize = cms.optional.untracked.bool
                ),
                appendToDataLabel = cms.string('hltCAGeometry'),
                caDCACuts = cms.vdouble(
                    0.05, 0.1, 0.1, 0.1, 0.1,
                    0.1, 0.1, 0.1, 0.1, 0.1
                ),
                caThetaCuts = cms.vdouble(
                    0.001, 0.001, 0.001, 0.001, 0.002,
                    0.002, 0.002, 0.002, 0.002, 0.002
                )
            )
        

    return process

def customizeHLTfor47630(process):
    attributes_to_remove = [
        'connectionRetrialPeriod',
        'connectionRetrialTimeOut',
        'connectionTimeOut',
        'enableConnectionSharing',
        'enablePoolAutomaticCleanUp',
        'enableReadOnlySessionOnUpdateConnection',
        'idleConnectionCleanupPeriod'
    ]

    for mod in modules_by_type(process, "PoolDBESSource"):
        if hasattr(mod, 'DBParameters'):
            pset = getattr(mod,'DBParameters')
            for attr in attributes_to_remove:
                if hasattr(pset, attr):
                    delattr(mod.DBParameters, attr)

    return process

# CMSSW version specific customizations
def customizeHLTforCMSSW(process, menuType="GRun"):

    process = customiseForOffline(process)

    # add call to action function in proper order: newest last!
    # process = customiseFor12718(process)
    
    process = customizeHLTfor47630(process)
    process = customizeHLTfor47611(process)

    return process


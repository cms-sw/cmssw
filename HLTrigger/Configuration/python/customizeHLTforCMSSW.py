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
        - cleans up the CANtupletAlpaka producers paramters;
        - adds the geometry paramters used to fill the CAGeometry;
        - adds the average sizes paramters to the CANtupletAlpaka producers;
        - for pp and HIN hlt setups.
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

            if not hasattr(prod, 'dzdrFact'):
                setattr(prod, 'dzdrFact', cms.double(8.0 * 0.0285 / 0.015))
            if not hasattr(prod, 'maxDYsize12'):
                setattr(prod, 'maxDYsize12', cms.int32(28))
            if not hasattr(prod, 'maxDYsize'):
                setattr(prod, 'maxDYsize', cms.int32(20))
            if not hasattr(prod, 'maxDYPred'):
                setattr(prod, 'maxDYPred', cms.int32(20))
            
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
            
            if not hasattr(prod, 'geometry'):

                geometryPS = cms.PSet(
                    startingPairs = cms.vuint32( [i for i in range(8)] + [i for i in range(13,20)]),
                    caDCACuts = cms.vdouble( [0.0918113099491] + [0.420724617835] * 9),
                    caThetaCuts = cms.vdouble([0.00123302705499] * 4 + [0.00355691321774] * 6),
                    pairGraph = cms.vuint32( 
                        0, 1, 0, 4, 0,
                        7, 1, 2, 1, 4,
                        1, 7, 4, 5, 7,
                        8, 2, 3, 2, 4,
                        2, 7, 5, 6, 8,
                        9, 0, 2, 1, 3,
                        0, 5, 0, 8, 
                        4, 6, 7, 9 
                    ),
                    phiCuts = cms.vint32( 
                        965, 1241, 395, 698, 1058,
                        1211, 348, 782, 1016, 810,
                        463, 755, 694, 531, 770,
                        471, 592, 750, 348
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
                        9., 9., 9., 9.)
                )
                
                setattr(prod, 'geometry', geometryPS)

    for ca_producer in ca_producers_hi:
        for prod in producers_by_type(process, ca_producer):
            
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

            if not hasattr(prod, 'geometry'):

                geometryPS = cms.PSet(
                    caDCACuts = cms.vdouble(
                        0.05, 0.1, 0.1, 0.1, 0.1,
                        0.1, 0.1, 0.1, 0.1, 0.1
                    ),
                    caThetaCuts = cms.vdouble(
                        0.001, 0.001, 0.001, 0.001, 0.002,
                        0.002, 0.002, 0.002, 0.002, 0.002
                    ),

                    ## This are the defaults actually
                    startingPairs = cms.vuint32(0,1,2),
                    pairGraph = cms.vuint32(0, 1, 0, 4, 0, 7, 1, 2, 1, 4, 1, 7, 4, 5, 7, 8, 2, 3, 2, 4, 2, 7, 5, 6, 8, 9, 0, 2, 1, 3, 0, 5, 0, 8, 4, 6, 7, 9),
                    phiCuts = cms.vint32(522, 730, 730, 522, 626, 626, 522, 522, 626, 626, 626, 522, 522, 522, 522, 522, 522, 522, 522),
                    minZ = cms.vdouble(-20, 0, -30, -22, 10, -30, -70, -70, -22, 15, -30, -70, -70, -20, -22, 0, -30, -70, -70),
                    maxZ = cms.vdouble(20, 30, 0, 22, 30, -10, 70, 70, 22, 30, -15, 70, 70, 20, 22, 30, 0, 70, 70),
                    maxR = cms.vdouble(20, 9, 9, 20, 7, 7, 5, 5, 20, 6, 6, 5, 5, 20, 20, 9, 9, 9, 9)
                )

                setattr(prod, 'geometry', geometryPS)

    return process

def customizeHLTfor48957(process):
    for filt in filters_by_type(process, "HLTL1NumberFilter"):
        if hasattr(filt, "fedId"):
            value = filt.fedId.value()
            del filt.fedId
            filt.fedIds = cms.vint32([value])

    return process

def customizeHLTfor48921(process):
    """ This customizer
        - renames some of the geometry parameters used to fill the CAGeometry:
            - minZ -> minInner
            - maxZ -> maxInner
            - maxR -> maxDR
        - for pp and HIN hlt setups.
    """
 
    ca_producers_pp = ['CAHitNtupletAlpakaPhase1@alpaka','alpaka_serial_sync::CAHitNtupletAlpakaPhase1']
    ca_producers_hi = ['CAHitNtupletAlpakaHIonPhase1@alpaka','alpaka_serial_sync::CAHitNtupletAlpakaHIonPhase1']
    ca_producers = ca_producers_pp + ca_producers_hi

    for ca_producer in ca_producers:
        for prod in producers_by_type(process, ca_producer):

            # cell pt cut
            if hasattr(prod, 'cellPtCut'):
                cellPtCut = getattr(prod, 'cellPtCut').value()
                delattr(prod, 'cellPtCut')
            else:
                cellPtCut = 0.85

            isPP = (ca_producer in ca_producers_pp)
                
            # geometry
            if hasattr(prod, 'geometry'):
                geometry = getattr(prod, "geometry")
            else:
                geometry = cms.PSet()

            # pair graph of layer pairs
            if not hasattr(geometry, "pairGraph"):
                setattr(geometry, "pairGraph", cms.vuint32( 
                    0, 1, 0, 4, 0,
                    7, 1, 2, 1, 4,
                    1, 7, 4, 5, 7,
                    8, 2, 3, 2, 4,
                    2, 7, 5, 6, 8,
                    9, 0, 2, 1, 3,
                    0, 5, 0, 8,
                    4, 6, 7, 9))
            
            nPairs = int(len(geometry.pairGraph) / 2)

            # dca cut for connections
            if not hasattr(geometry, "caDCACuts"):
                if isPP:
                    setattr(geometry, "caDCACuts", cms.vdouble([0.0918113099491] + [0.420724617835] * 9))
                else:
                    setattr(geometry, "caDCACuts", cms.vdouble(0.05, 0.1, 0.1, 0.1, 0.1,
                                                                0.1, 0.1, 0.1, 0.1, 0.1))

            nLayers = len(geometry.caDCACuts)
            
            # theta cut for connections
            if not hasattr(geometry, "caThetaCuts"):
                if isPP:
                    setattr(geometry, "caThetaCuts", cms.vdouble([0.00123302705499] * 4 + [0.00355691321774] * 6))
                else:
                    setattr(geometry, "caThetaCuts", cms.vdouble(0.001, 0.001, 0.001, 0.001, 0.002,
                                                                 0.002, 0.002, 0.002, 0.002, 0.002))
            
            # is barrel information for layers (set to true for all layers to force min/maxInner cuts to act in z direction)
            if not hasattr(geometry, "isBarrel"):
                setattr(geometry, "isBarrel", cms.vint32([1] * nLayers))

            if not hasattr(geometry, "startingPairs"):
                if isPP:
                    setattr(geometry, "startingPairs", cms.vuint32( [i for i in range(8)] + [i for i in range(13,20)]))
                else:
                    setattr(geometry, "startingPairs", cms.vuint32(0,1,2))

            # delta phi cuts
            if not hasattr(geometry, "phiCuts"):
                if isPP:
                    setattr(geometry, "phiCuts", cms.vint32( 
                            965, 1241, 395, 698, 1058,
                            1211, 348, 782, 1016, 810,
                            463, 755, 694, 531, 770,
                            471, 592, 750, 348))
                else:
                    setattr(geometry, "phiCuts", cms.vint32( 
                            522, 730, 730, 522, 626, 626, 522, 522, 626, 626, 626, 522, 522, 522, 522, 522, 522, 522, 522))

            # minInner 
            if hasattr(geometry, "minZ"):
                minZ = getattr(geometry, "minZ")
                delattr(geometry, "minZ")
            else:
                minZ = cms.vdouble(-20, 0, -30, -22, 10, -30, -70, -70, -22, 15, -30, -70, -70, -20, -22, 0, -30, -70, -70)
            if not hasattr(geometry, "minInner"):
                setattr(geometry, "minInner", minZ)

            # maxInner      
            if hasattr(geometry, "maxZ"):
                maxZ = getattr(geometry, "maxZ")
                delattr(geometry, "maxZ")
            else:
                maxZ = cms.vdouble(20, 30, 0, 22, 30, -10, 70, 70, 22, 30, -15, 70, 70, 20, 22, 30, 0, 70, 70)
            if not hasattr(geometry, "maxInner"):
                setattr(geometry, "maxInner", maxZ)

            # minOuter
            if not hasattr(geometry, "minOuter"):
                setattr(geometry, "minOuter", cms.vdouble([-10000] * nPairs))
            
            # maxOuter
            if not hasattr(geometry, "maxOuter"):
                setattr(geometry, "maxOuter", cms.vdouble([ 10000] * nPairs))

            # maxDR
            if hasattr(geometry, "maxR"):
                maxR = getattr(geometry, "maxR")
                delattr(geometry, "maxR")
            else:
                maxR = cms.vdouble(20, 9, 9, 20, 7, 7, 5, 5, 20, 6, 6, 5, 5, 20, 20, 9, 9, 9, 9)
            if not hasattr(geometry, "maxDR"):
                setattr(geometry, "maxDR", maxR)

            # minDZ
            if not hasattr(geometry, "minDZ"):
                setattr(geometry, "minDZ", cms.vdouble([-10000] * nPairs))

            if not hasattr(geometry, "maxDZ"):
                setattr(geometry, "maxDZ", cms.vdouble([ 10000] * nPairs))
            
            if not hasattr(geometry, "ptCuts"):
                setattr(geometry, "ptCuts", cms.vdouble([ cellPtCut] * nPairs))
                
            # set the full geometry
            setattr(prod, 'geometry', geometry)

    return process

# CMSSW version specific customizations
def customizeHLTforCMSSW(process, menuType="GRun"):

    process = customiseForOffline(process)

    # add call to action function in proper order: newest last!
    # process = customiseFor12718(process)
    
    process = customizeHLTfor47611(process)
    process = customizeHLTfor48957(process)
    process = customizeHLTfor48921(process)

    return process

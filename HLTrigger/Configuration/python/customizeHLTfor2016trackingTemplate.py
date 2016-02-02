import FWCore.ParameterSet.Config as cms

#####################################################################################
### this customization is meant for providing the tracking template for 2016
### it has 3 main parts
### 0. bug fix in the MeasurementTrackerESProducer module
### 1. new selectors 
###    (AnalyticalTrackSelector --> TrackCutClassifier + TrackCollectionFilterCloner)
###    by using this new track selector modules,
###    we have to slightly modify the sequence
### 2. new CCC
###    in order to cope w/ the strip hit inefficiency we discovered in 2015,
###    we have to slightly modify the strip CCC
###    - decrease the CC threshold @ building step 
###      (HLTSiStripClusterChargeCutLoose --> HLTSiStripClusterChargeCutTiny)
###    - add a new CCC @ filter step for limiting the timing and the fakerate
###      for CC between HLTSiStripClusterChargeCutTiny and HLTSiStripClusterChargeCutLoose
###      allow missing hits (by using new maxCCCLostHits paraemter)
### 3. speed up
###    @building step limit MaxDisplacement, MaxSagitta and MinimalTolerance
###    @filter step make use of seedExtension
#####################################################################################


# reusable functions
def producers_by_type(process, *types):
    return (module for module in process._Process__producers.values() if module._TypedParameterizable__type in types)

def esproducers_by_type(process, *types):
    return (module for module in process._Process__esproducers.values() if module._TypedParameterizable__type in types)

def bug_fix(process):
    # fix 2015 bug
    for module in esproducers_by_type(process, 'MeasurementTrackerESProducer'):
        module.badStripCuts.TOB = cms.PSet(
          maxConsecutiveBad = cms.uint32( 2 ),
          maxBad = cms.uint32( 4 )
        )
        module.badStripCuts.TIB = cms.PSet(
          maxConsecutiveBad = cms.uint32( 2 ),
          maxBad = cms.uint32( 4 )
        )
        module.badStripCuts.TID = cms.PSet(
          maxConsecutiveBad = cms.uint32( 2 ),
          maxBad = cms.uint32( 4 )
        )
        module.badStripCuts.TEC = cms.PSet(
          maxConsecutiveBad = cms.uint32( 2 ),
          maxBad = cms.uint32( 4 )
        )
    return process

def CCC(process):

    # new CCC
    setattr(process,'HLTSiStripClusterChargeCutTiny', cms.PSet(  value = cms.double(  800.0 ) ) )
    if hasattr(process,'hltESPChi2ChargeMeasurementEstimator16'): # used by iter1,2,3,4
	getattr(process,'hltESPChi2ChargeMeasurementEstimator16').clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutTiny" ) ) # 2015 HLTSiStripClusterChargeCutLoose
    if hasattr(process,'hltESPChi2ChargeMeasurementEstimator9'): # used by iter0
	getattr(process,'hltESPChi2ChargeMeasurementEstimator9').clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutTiny" ) ) # 2015 HLTSiStripClusterChargeCutLoose

    if hasattr(process, 'HLTIter0PSetTrajectoryFilterIT'):
        getattr(process,'HLTIter0PSetTrajectoryFilterIT').minGoodStripCharge = cms.PSet(refToPSet_ = cms.string('HLTSiStripClusterChargeCutLoose')) # default HLTSiStripClusterChargeCutNone
        getattr(process,'HLTIter0PSetTrajectoryFilterIT').maxCCCLostHits      = cms.int32(1)
    if hasattr(process, 'HLTIter1PSetTrajectoryFilterIT'):
        getattr(process,'HLTIter1PSetTrajectoryFilterIT').minGoodStripCharge  = cms.PSet(refToPSet_ = cms.string('HLTSiStripClusterChargeCutLoose')) # default HLTSiStripClusterChargeCutNone
        getattr(process,'HLTIter1PSetTrajectoryFilterIT').maxCCCLostHits      = cms.int32(1)
    if hasattr(process, 'HLTIter2PSetTrajectoryFilterIT'):
        getattr(process,'HLTIter2PSetTrajectoryFilterIT').minGoodStripCharge  = cms.PSet(refToPSet_ = cms.string('HLTSiStripClusterChargeCutLoose')) # default HLTSiStripClusterChargeCutNone
        getattr(process,'HLTIter2PSetTrajectoryFilterIT').maxCCCLostHits      = cms.int32(1)
    if hasattr(process, 'HLTIter3PSetTrajectoryFilterIT'):
        getattr(process,'HLTIter3PSetTrajectoryFilterIT').minGoodStripCharge  = cms.PSet(refToPSet_ = cms.string('HLTSiStripClusterChargeCutLoose')) # default HLTSiStripClusterChargeCutNone
        getattr(process,'HLTIter3PSetTrajectoryFilterIT').maxCCCLostHits      = cms.int32(1)
    if hasattr(process, 'HLTIter4PSetTrajectoryFilterIT'):
        getattr(process,'HLTIter4PSetTrajectoryFilterIT').minGoodStripCharge  = cms.PSet(refToPSet_ = cms.string('HLTSiStripClusterChargeCutLoose')) # default HLTSiStripClusterChargeCutNone
        getattr(process,'HLTIter4PSetTrajectoryFilterIT').maxCCCLostHits      = cms.int32(1)

    return process

def speedup_building(process):
    # speed up
    process.HLTSiStripClusterChargeCutTiny = cms.PSet(  value = cms.double(  800.0 ) )    
    if hasattr(process,'hltESPChi2ChargeMeasurementEstimator16'): # used by iter1,2,3,4
	getattr(process,'hltESPChi2ChargeMeasurementEstimator16').MaxDisplacement  = cms.double(0.5) # default 100
	getattr(process,'hltESPChi2ChargeMeasurementEstimator16').MaxSagitta       = cms.double(2)   # default -1
	getattr(process,'hltESPChi2ChargeMeasurementEstimator16').MinimalTolerance = cms.double(0.5) # default 10
 
    if hasattr(process,'hltESPChi2ChargeMeasurementEstimator9'): # used by iter0
	getattr(process,'hltESPChi2ChargeMeasurementEstimator9').MaxDisplacement  = cms.double(0.5) # default 100
	getattr(process,'hltESPChi2ChargeMeasurementEstimator9').MaxSagitta       = cms.double(2)   # default -1
	getattr(process,'hltESPChi2ChargeMeasurementEstimator9').MinimalTolerance = cms.double(0.5) # default 10

    return process

def speedup_filtering(process):
    # speed up
    if hasattr(process, 'HLTIter0PSetTrajectoryFilterIT'):
        getattr(process,'HLTIter0PSetTrajectoryFilterIT').seedExtension       = cms.int32(0)
        getattr(process,'HLTIter0PSetTrajectoryFilterIT').strictSeedExtension = cms.bool(False)
    if hasattr(process, 'HLTIter1PSetTrajectoryFilterIT'):
        getattr(process,'HLTIter1PSetTrajectoryFilterIT').seedExtension       = cms.int32(0)
        getattr(process,'HLTIter1PSetTrajectoryFilterIT').strictSeedExtension = cms.bool(False)
    if hasattr(process, 'HLTIter2PSetTrajectoryFilterIT'):
        getattr(process,'HLTIter2PSetTrajectoryFilterIT').seedExtension       = cms.int32(1)
        getattr(process,'HLTIter2PSetTrajectoryFilterIT').strictSeedExtension = cms.bool(False)
    if hasattr(process, 'HLTIter3PSetTrajectoryFilterIT'):
        getattr(process,'HLTIter3PSetTrajectoryFilterIT').seedExtension       = cms.int32(0)
        getattr(process,'HLTIter3PSetTrajectoryFilterIT').strictSeedExtension = cms.bool(False)
    if hasattr(process, 'HLTIter4PSetTrajectoryFilterIT'):
        getattr(process,'HLTIter4PSetTrajectoryFilterIT').seedExtension       = cms.int32(0)
        getattr(process,'HLTIter4PSetTrajectoryFilterIT').strictSeedExtension = cms.bool(False)

    return process


def new_selector(process):

    # new selectors
    # iter0
    if hasattr(process, 'HLTIterativeTrackingIteration0'):

        setattr(process,'hltIter0PFlowTrackCutClassifier', cms.EDProducer('TrackCutClassifier',
           src = cms.InputTag('hltIter0PFlowCtfWithMaterialTracks'),
           beamspot = cms.InputTag('hltOnlineBeamSpot'),
           vertices = cms.InputTag('hltTrimmedPixelVertices'),
           GBRForestLabel = cms.string(''),
           GBRForestFileName = cms.string(''),
           qualityCuts = cms.vdouble(
            -0.7,
             0.1,
             0.7
           ),
           mva = cms.PSet(
             minNdof = cms.vdouble(
               1E-5,
               1E-5,
               1E-5
             ),
             minPixelHits = cms.vint32(
               0,
               0,
               0
             ),
             minLayers = cms.vint32(
               3,
               3,
               3
             ),
             min3DLayers = cms.vint32(
               0,
               0,
               0
             ),
             maxLostLayers = cms.vint32(
               1,
               1,
               1
             ),
             maxChi2 = cms.vdouble(
               9999,
               25.,
               16.
             ),
             maxChi2n = cms.vdouble(
               1.2,
               1.,
               0.7
             ),
             minNVtxTrk = cms.int32(3),
             maxDz = cms.vdouble(
               0.5,
               0.2,
         #      0.2
               3.4028234663852886e+38
             ),
             maxDzWrtBS = cms.vdouble(
               3.4028234663852886e+38,
               24,
               15
             ),
             maxDr = cms.vdouble(
               0.5,
               0.03,
         #      0.03
               3.4028234663852886e+38
             ),
             dz_par = cms.PSet(
                     dz_exp = cms.vint32(
                         4,
                         4,
                         4,
                     ),
                     dz_par1 = cms.vdouble(
                         0.4,
                         0.4,
                         0.4
         #                3.4028234663852886e+38
                     ),
                     dz_par2 = cms.vdouble(
                         0.35,
                         0.35,
                         0.35
         #                3.4028234663852886e+38
                     ),
             ),
             dr_par = cms.PSet(
                     dr_exp = cms.vint32(
                         4,
                         4,
                         4,
                     ),
                     dr_par1 = cms.vdouble(
                         0.4,
                         0.4,
                         0.4
         #                3.4028234663852886e+38
                     ),
                     dr_par2 = cms.vdouble(
                         0.3,
                         0.3,
                         0.3
         #                3.4028234663852886e+38
                     ),
               d0err = cms.vdouble(
                 0.003,
                 0.003,
                 0.003
               ),
               d0err_par = cms.vdouble(
                 0.001,
                 0.001,
                 0.001
               )
             )
           )
         )
        )        
        setattr(process,'hltIter0PFlowTrackSelectionHighPurity', cms.EDProducer( "TrackCollectionFilterCloner",
          originalSource   = cms.InputTag('hltIter0PFlowCtfWithMaterialTracks'),
          originalMVAVals  = cms.InputTag('hltIter0PFlowTrackCutClassifier','MVAValues'),
          originalQualVals = cms.InputTag('hltIter0PFlowTrackCutClassifier','QualityMasks'),
          minQuality       = cms.string('highPurity') ,
          cloner = cms.untracked.PSet(
            copyExtras       = cms.untracked.bool(False),
            copyTrajectories = cms.untracked.bool(False)
          )
         )
        )
        iter0HP  = getattr(process,'hltIter0PFlowTrackSelectionHighPurity')

        iter0seq = getattr(process,'HLTIterativeTrackingIteration0')
        iter0seq.insert( iter0seq.index( iter0HP ), getattr(process,'hltIter0PFlowTrackCutClassifier') )
#        iter02seq = getattr(process,'HLTIterativeTrackingIter02')
#        iter02seq.insert( iter02seq.index( iter0HP ), process.hltIter0PFlowTrackCutClassifier )
        

    ### iter1
    if hasattr(process, 'HLTIterativeTrackingIteration1'):

        setattr(process,'hltIter1PFlowTrackCutClassifierPrompt', cms.EDProducer('TrackCutClassifier',
           src = cms.InputTag('hltIter1PFlowCtfWithMaterialTracks'),
           beamspot = cms.InputTag('hltOnlineBeamSpot'),
           vertices = cms.InputTag('hltTrimmedPixelVertices'),
           GBRForestLabel = cms.string(''),
           GBRForestFileName = cms.string(''),
           qualityCuts = cms.vdouble(
            -0.7,
             0.1,
             0.7
           ),
           mva = cms.PSet(
             minNdof = cms.vdouble(
               1E-5,
               1E-5,
               1E-5
             ),
             minPixelHits = cms.vint32(
               0,
               0,
               2
             ),
             minLayers = cms.vint32(
               3,
               3,
               3
             ),
             min3DLayers = cms.vint32(
               0,
               0,
               0
             ),
             maxLostLayers = cms.vint32(
               1,
               1,
               1
             ),
             maxChi2 = cms.vdouble(
               9999.,
               25.,
               16.
             ),
             maxChi2n = cms.vdouble(
               1.2,
               1.,
               0.7
             ),
             minNVtxTrk = cms.int32(3),
             maxDz = cms.vdouble(
               3.4028234663852886e+38,
               1.0,
        #       0.4
               3.4028234663852886e+38
             ),
             maxDzWrtBS = cms.vdouble(
               3.4028234663852886e+38,
               24,
               15
             ),
             maxDr = cms.vdouble(
               3.4028234663852886e+38,
               1.0,
        #       0.1
               3.4028234663852886e+38
             ),
             dz_par = cms.PSet(
                     dz_exp = cms.vint32(
                         3,
                         3,
                         3,
                     ),
                     dz_par1 = cms.vdouble(
                         3.4028234663852886e+38,
                         1.0,
                         0.9
        #                 3.4028234663852886e+38
                     ),
                     dz_par2 = cms.vdouble(
                         3.4028234663852886e+38,
                         1.0,
                         0.8
        #                 3.4028234663852886e+38
                     ),
             ),
             dr_par = cms.PSet(
                     dr_exp = cms.vint32(
                         3,
                         3,
                         3,
                     ),
                     dr_par1 = cms.vdouble(
                         3.4028234663852886e+38,
                         1.0,
                         0.9
        #                 3.4028234663852886e+38
                     ),
                     dr_par2 = cms.vdouble(
                         3.4028234663852886e+38,
                         1.0,
                         0.85
        #                 3.4028234663852886e+38
                     ),
               d0err = cms.vdouble(
                 0.003,
                 0.003,
                 0.003
               ),
               d0err_par = cms.vdouble(
                 0.001,
                 0.001,
                 0.001
               )
             )
           )
        )
        )
        
        setattr(process,'hltIter1PFlowTrackCutClassifierDetached', cms.EDProducer('TrackCutClassifier',
          src = cms.InputTag('hltIter1PFlowCtfWithMaterialTracks'),
          beamspot = cms.InputTag('hltOnlineBeamSpot'),
          vertices = cms.InputTag('hltTrimmedPixelVertices'),
          GBRForestLabel = cms.string(''),
          GBRForestFileName = cms.string(''),
          qualityCuts = cms.vdouble(
           -0.7,
            0.1,
            0.7
          ),
          mva = cms.PSet(
            minNdof = cms.vdouble(
              1E-5,
              1E-5,
              1E-5
            ),
            minPixelHits = cms.vint32(
              0,
              0,
              2
            ),
            minLayers = cms.vint32(
              5,
              5,
              5
            ),
            min3DLayers = cms.vint32(
              0,
              0,
              0
            ),
            maxLostLayers = cms.vint32(
              1,
              1,
              1
            ),
            maxChi2 = cms.vdouble(
              9999.,
              25.,
              16.
            ),
            maxChi2n = cms.vdouble(
              1.0,
              0.7,
              0.4
            ),
            minNVtxTrk = cms.int32(3),
            maxDz = cms.vdouble(
              3.4028234663852886e+38,
              1.0,
        #      0.5
              3.4028234663852886e+38
            ),
            maxDzWrtBS = cms.vdouble(
              3.4028234663852886e+38,
              24,
              15
            ),
            maxDr = cms.vdouble(
              3.4028234663852886e+38,
              1.0,
        #      0.2
              3.4028234663852886e+38
            ),
            dz_par = cms.PSet(
                    dz_exp = cms.vint32(
                        4,
                        4,
                        4,
                    ),
                    dz_par1 = cms.vdouble(
                        1.0,
                        1.0,
                        1.0
        #                3.4028234663852886e+38
                    ),
                    dz_par2 = cms.vdouble(
                        1.0,
                        1.0,
                        1.0
        #                3.4028234663852886e+38
                    ),
            ),
            dr_par = cms.PSet(
                    dr_exp = cms.vint32(
                        4,
                        4,
                        4,
                    ),
                    dr_par1 = cms.vdouble(
                        1.0,
                        1.0,
                        1.0
        #                3.4028234663852886e+38
                    ),
                    dr_par2 = cms.vdouble(
                        1.0,
                        1.0,
                        1.0
        #                3.4028234663852886e+38
                    ),
              d0err = cms.vdouble(
                0.003,
                0.003,
                0.003
              ),
              d0err_par = cms.vdouble(
                0.001,
                0.001,
                0.001
              )
        
            )
          )
         )
        )
        setattr(process,'hltIter1PFlowTrackCutClassifierMerged', cms.EDProducer('ClassifierMerger',
          inputClassifiers = cms.vstring('hltIter1PFlowTrackCutClassifierPrompt','hltIter1PFlowTrackCutClassifierDetached')
         )
        )
          
        setattr(process,'hltIter1PFlowTrackSelectionHighPurity', cms.EDProducer( "TrackCollectionFilterCloner",
          originalSource   = cms.InputTag('hltIter1PFlowCtfWithMaterialTracks'),
          originalMVAVals  = cms.InputTag('hltIter1PFlowTrackCutClassifierMerged','MVAValues'),
          originalQualVals = cms.InputTag('hltIter1PFlowTrackCutClassifierMerged','QualityMasks'),
          minQuality       = cms.string('highPurity') ,
          cloner = cms.untracked.PSet(
            copyExtras       = cms.untracked.bool(False),
            copyTrajectories = cms.untracked.bool(False)
          )
         )
        )

        iter1prompt   = getattr(process,'hltIter1PFlowTrackCutClassifierPrompt')
        iter1detached = getattr(process,'hltIter1PFlowTrackCutClassifierDetached')
        iter1merge    = getattr(process,'hltIter1PFlowTrackCutClassifierMerged')
        iter1HP       = getattr(process,'hltIter1PFlowTrackSelectionHighPurity')

        iter1seq = getattr(process,'HLTIterativeTrackingIteration1')
        iter1seq.replace( getattr(process,'hltIter1PFlowTrackSelectionHighPurityLoose'), iter1prompt )
        iter1seq.replace( getattr(process,'hltIter1PFlowTrackSelectionHighPurityTight'), iter1detached )
        iter1seq.insert( iter1seq.index( iter1HP ), iter1merge )

        iter02seq = getattr(process,'HLTIterativeTrackingIter02')
        iter02seq.replace( getattr(process,'hltIter1PFlowTrackSelectionHighPurityLoose'), iter1prompt )
        iter02seq.replace( getattr(process,'hltIter1PFlowTrackSelectionHighPurityTight'), iter1detached )
        iter02seq.insert( iter02seq.index( iter1HP ), iter1merge )

    #### iter2
    if hasattr(process, 'HLTIterativeTrackingIteration2'):

        setattr(process,'hltIter2PFlowTrackCutClassifier', cms.EDProducer('TrackCutClassifier',
          src = cms.InputTag('hltIter2PFlowCtfWithMaterialTracks'),
          beamspot = cms.InputTag('hltOnlineBeamSpot'),
          vertices = cms.InputTag('hltTrimmedPixelVertices'),
          GBRForestLabel = cms.string(''),
          GBRForestFileName = cms.string(''),
          qualityCuts = cms.vdouble(
           -0.7,
            0.1,
            0.7
          ),
          mva = cms.PSet(
            minNdof = cms.vdouble(
              1E-5,
              1E-5,
              1E-5
            ),
            minPixelHits = cms.vint32(
              0,
              0,
              0
            ),
            minLayers = cms.vint32(
              3,
              3,
              3
            ),
            min3DLayers = cms.vint32(
              0,
              0,
              0
            ),
            maxLostLayers = cms.vint32(
              1,
              1,
              1
            ),
            maxChi2 = cms.vdouble(
              9999,
              25.,
              16.
            ),
            maxChi2n = cms.vdouble(
              1.2,
              1.,
              0.7
            ),
            minNVtxTrk = cms.int32(3),
            maxDz = cms.vdouble(
              0.5,
              0.2,
              3.4028234663852886e+38,
        #      0.1 # 0.2
            ),
            maxDzWrtBS = cms.vdouble(
              3.4028234663852886e+38,
              24,
              15
            ),
            maxDr = cms.vdouble(
              0.5,
              0.03,
              3.4028234663852886e+38,
        #      0.03
            ),
            dz_par = cms.PSet(
                    dz_exp = cms.vint32(
                        4,
                        4,
                        4,
                    ),
                    dz_par1 = cms.vdouble(
                        3.4028234663852886e+38,
                        0.4,
                        0.4
                    ),
                    dz_par2 = cms.vdouble(
                        3.4028234663852886e+38,
                        0.35,
                        0.35
                    ),
            ),
            dr_par = cms.PSet(
                    dr_exp = cms.vint32(
                        4,
                        4,
                        4,
                    ),
                    dr_par1 = cms.vdouble(
                        3.4028234663852886e+38,
                        0.4,
                        0.4
                    ),
                    dr_par2 = cms.vdouble(
                        3.4028234663852886e+38,
                        0.3,
                        0.3
                    ),
              d0err = cms.vdouble(
                0.003,
                0.003,
                0.003
              ),
              d0err_par = cms.vdouble(
                0.001,
                0.001,
                0.001
              )
            )
          )
         )
        )
        setattr(process,'hltIter2PFlowTrackSelectionHighPurity', cms.EDProducer( "TrackCollectionFilterCloner",
          originalSource   = cms.InputTag('hltIter2PFlowCtfWithMaterialTracks'),
          originalMVAVals  = cms.InputTag('hltIter2PFlowTrackCutClassifier','MVAValues'),
          originalQualVals = cms.InputTag('hltIter2PFlowTrackCutClassifier','QualityMasks'),
          minQuality       = cms.string('highPurity') ,
          cloner = cms.untracked.PSet(
            copyExtras       = cms.untracked.bool(False),
            copyTrajectories = cms.untracked.bool(False)
          )
         )
        )
        iter2HP  = getattr(process,'hltIter2PFlowTrackSelectionHighPurity')

        iter2seq = getattr(process,'HLTIterativeTrackingIteration2')
        iter2seq.insert( iter2seq.index( iter2HP ), getattr(process,'hltIter2PFlowTrackCutClassifier') )
        iter02seq = getattr(process,'HLTIterativeTrackingIter02')
        if hasattr(iter02seq,'hltIter2PFlowTrackSelectionHighPurity'):
            iter02seq.insert( iter02seq.index( iter2HP ), getattr(process,'hltIter2PFlowTrackCutClassifier') )

    return process

def customiseFor2016trackingTemplate(process):
    process = bug_fix(process)
    process = new_selector(process)
    
    process = CCC(process)
    process = speedup_building(process)
    process = speedup_filtering(process)

    return process


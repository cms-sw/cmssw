import FWCore.ParameterSet.Config as cms

def memorySavingTracking(process):
    toRemove={}
    #list of modules in iterative tracking
    ## remove rekeying of clusters refs from track producer
    trackProducers=['preFilterZeroStepTracks',
                    'preFilterStepOneTracks',
                    'secWithMaterialTracks',
                    'thWithMaterialTracks',
                    'fourthWithMaterialTracks',
                    'fifthWithMaterialTracks',
                    ]

    for tp in trackProducers:
        m=getattr(process,tp)
        if hasattr(m,"clusterRemovalInfo"):
            #print "removing cluter rekeying from",tp
            delattr(m,"clusterRemovalInfo")

    measurementTrackers=['MeasurementTracker',
                         'newMeasurementTracker',
                         'secMeasurementTracker',
                         'thMeasurementTracker',
                         'fourthMeasurementTracker',
                         'fifthMeasurementTracker',
                         ]

    # list of measurement tracker, component names
    ## create the clusterRef to skip creators (MT+'ToSkip')
    for mt in measurementTrackers:
        es=getattr(process,mt)
        ## modify MT to point to the full cluster list
        #if es.pixelClusterProducer.value() == 'siPixelClusters':
        #    continue

        #old trackclusterremoval module
        removalModule=es.pixelClusterProducer.value()

        if (removalModule != 'siPixelClusters'):
            es.skipClusters = cms.InputTag(removalModule)
            es.pixelClusterProducer = 'siPixelClusters'
            es.stripClusterProducer = 'siStripClusters'
            #print mt,es.skipClusters,es.pixelClusterProducer,es.stripClusterProducer
            tcremoval = getattr(process,removalModule)
            #print removalModule,"turned to using new scheme"
            tcremoval.clusterLessSolution= cms.bool(True)
            tcremoval.stripClusters = 'siStripClusters'
            tcremoval.pixelClusters = 'siPixelClusters'
            skipTrackQualityFilter=False
            if (skipTrackQualityFilter):
                tcremoval.TrackQuality = cms.string('highPurity')
                #remove the QualityFilter module from the path
                toRemove[tcremoval.trajectories.value()]=True
                qf=getattr(process,tcremoval.trajectories.value())
                tcremoval.trajectories = qf.recTracks
        #else:
            #print mt,'no cluster to skip',es.pixelClusterProducer,es.stripClusterProducer

    patternRecoModules=[
        'fifthTrackCandidates',
        'fourthTrackCandidates',
        'newTrackCandidateMaker',
        'secTrackCandidates',
        'stepOneTrackCandidateMaker',
        'thTrackCandidates'
        ]
    
    for ckfm in patternRecoModules:
        ckf=getattr(process,ckfm)
        builder=getattr(process,ckf.TrajectoryBuilder.value())
        mtn= builder.MeasurementTrackerName.value()
        if mtn!='':
            #make it look at the central MT
            builder.MeasurementTrackerName=''
            mt=getattr(process,mtn)
            # transfer the cluster removal from the MT to the builder
            builder.clustersToSkip = mt.skipClusters
            #print "setting",ckf.TrajectoryBuilder.value(),"via",ckfm,"to look at central MT"
            #print "removing MT:",mtn
            delattr(process,mtn)
        #else:
            #print ckfm,"untouched"
            
    #all seeding layers should point to the same rechits collections
    for esp in process.es_producers_().keys():
        es = getattr(process,esp)
        if es._TypedParameterizable__type != 'SeedingLayersESProducer':
            continue
        for pm in es.parameters_().keys():
            p=getattr(es,pm)
            if p.pythonTypeName() == 'cms.PSet':
                if hasattr(p,'HitProducer'):
                    #print "pixel",pm,p
                    #pixel case
                    if p.HitProducer != 'siPixelRecHits':
                        toRemove[p.HitProducer.value()]=True
                        skip=getattr(process,p.HitProducer.value()).src
                        p.HitProducer = 'siPixelRecHits'
                        #and set the skipping
                        p.skipClusters = cms.InputTag(skip.value())
                        #print esp,"modified for new skipping"
                        #print esp,pm,p

                if hasattr(p,'matchedRecHits'):
                    #print "strip",pm,p
                    #strip case
                    ## rename the collection
                    if p.matchedRecHits.moduleLabel != 'siStripMatchedRecHits':
                        toRemove[p.matchedRecHits.moduleLabel]=True
                        skip=getattr(process,p.matchedRecHits.moduleLabel).ClusterProducer
                        p.matchedRecHits.setModuleLabel('siStripMatchedRecHits')
                        #and set the skipping
                        p.skipClusters = cms.InputTag(skip.value())
                        #print esp,pm,p


    for edp in process.producers_():
        p=getattr(process,edp)
        if hasattr(p,'ClusterCheckPSet'):
            #print "resetting cluster check for",edp
            p.ClusterCheckPSet.PixelClusterCollectionLabel = 'siPixelClusters'
            p.ClusterCheckPSet.ClusterCollectionLabel = 'siStripClusters'

    #force useless module to be removed
    toRemove['secStripRecHits']=True
    toRemove['fourthPixelRecHits']=True
    toRemove['fifthPixelRecHits']=True
    
    for tr in toRemove:
        if hasattr(process,tr):
            #print "removing",tr
            process.reconstruction_step.remove(getattr(process,tr))

    delattr(process.newCombinedSeeds,'clusterRemovalInfos')
        
    return (process)


def customiseCommon(process):

    process = memorySavingTracking(process)
    
    return (process)


##############################################################################
def customisePPData(process):
    process= customiseCommon(process)

    ## particle flow HF cleaning
    process.particleFlowRecHitHCAL.LongShortFibre_Cut = 30.
    process.particleFlowRecHitHCAL.ApplyPulseDPG = True

    ## HF cleaning for data only
    process.hcalRecAlgos.SeverityLevels[3].RecHitFlags.remove("HFDigiTime")
    process.hcalRecAlgos.SeverityLevels[4].RecHitFlags.append("HFDigiTime")

    ##beam-halo-id for data only
    process.CSCHaloData.ExpectedBX = cms.int32(3)

    ## hcal hit flagging
    process.hfreco.PETstat.flagsToSkip  = 2
    process.hfreco.S8S1stat.flagsToSkip = 18
    process.hfreco.S9S1stat.flagsToSkip = 26

    return process


##############################################################################
def customisePPMC(process):
    process=customiseCommon(process)
    
    return process

##############################################################################
def customiseCosmicData(process):

    return process

##############################################################################
def customiseCosmicMC(process):
    
    return process
        
##############################################################################
def customiseVALSKIM(process):
    process= customisePPData(process)
    process.reconstruction.remove(process.lumiProducer)
    return process
                
##############################################################################
def customiseExpress(process):
    process= customisePPData(process)

    import RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi
    process.offlineBeamSpot = RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi.onlineBeamSpotProducer.clone()
    
    return process

##############################################################################
def customisePrompt(process):
    process= customisePPData(process)
    return process

##############################################################################
##############################################################################

def customiseCommonHI(process):
    
    ###############################################################################################
    ####
    ####  Top level replaces for handling strange scenarios of early HI collisions
    ####

    ## Offline Silicon Tracker Zero Suppression
    process.siStripZeroSuppression.Algorithms.CommonModeNoiseSubtractionMode = cms.string("IteratedMedian")
    process.siStripZeroSuppression.Algorithms.CutToAvoidSignal = cms.double(2.0)
    process.siStripZeroSuppression.Algorithms.Iterations = cms.int32(3)
    process.siStripZeroSuppression.storeCM = cms.bool(True)


    ###
    ###  end of top level replacements
    ###
    ###############################################################################################

    return process

##############################################################################
def customiseExpressHI(process):
    process= customiseCommonHI(process)

    import RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi
    process.offlineBeamSpot = RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi.onlineBeamSpotProducer.clone()
    
    return process

##############################################################################
def customisePromptHI(process):
    process= customiseCommonHI(process)

    import RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi
    process.offlineBeamSpot = RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi.onlineBeamSpotProducer.clone()
    
    return process

##############################################################################

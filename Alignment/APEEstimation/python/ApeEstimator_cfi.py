import os

import FWCore.ParameterSet.Config as cms


ApeEstimatorTemplate = cms.EDAnalyzer('ApeEstimator',
    
    #Input source of Tracks
    tjTkAssociationMapTag = cms.InputTag("TrackRefitterForApeEstimator"),
    
    #Max no. of tracks per event:
    # default = 0, no event selection
    maxTracksPerEvent = cms.uint32(0),
    
    #Perform Track Cuts
    applyTrackCuts = cms.bool(True),
    
    # Selection of useful hits for analysis
    HitSelector = cms.PSet(
      # FIXME: create own PSets for Pixel and Strip?
      
      #Parameters for Cuts on Strip Clusters (independent of track reconstruction, but associated to a track's hit)
      width = cms.vuint32(),        #interval, needs even number of arguments. for int specify one number n as interval (n,n)
      widthProj = cms.vdouble(),
      widthDiff = cms.vdouble(),
      charge = cms.vdouble(),
      edgeStrips = cms.vuint32(),   #how many strips on edge to exclude wrt. maxStrip (on both edges)
      maxCharge = cms.vdouble(),
      chargeOnEdges = cms.vdouble(),    # fraction of charge on edge strips of cluster
      chargeAsymmetry = cms.vdouble(),     # asymmetry of charge on edge strips of cluster
      chargeLRplus = cms.vdouble(),    # fraction of charge left and right from strip with maxCharge
      chargeLRminus = cms.vdouble(),     # asymmetry of charge left and right from strip with maxCharge
      maxIndex = cms.vuint32(),
      sOverN = cms.vdouble(),
      
      #Parameters for Cuts on Pixel Clusters (independent of track reconstruction, but associated to a track's hit)
      chargePixel = cms.vdouble(),
      widthX = cms.vuint32(),
      widthY = cms.vuint32(),
      baryStripX = cms.vdouble(),
      baryStripY = cms.vdouble(),
      clusterProbabilityXY = cms.vdouble(),
      clusterProbabilityQ = cms.vdouble(),
      clusterProbabilityXYQ = cms.vdouble(),
      logClusterProbability = cms.vdouble(),
      isOnEdge = cms.vuint32(),
      hasBadPixels = cms.vuint32(),
      spansTwoRoc = cms.vuint32(),
      qBin = cms.vuint32(),
      
      #Parameters for Cuts on Pixel+Strip Hits (depending on track reconstruction)
      phiSens = cms.vdouble(), #trajectory angle on module
      phiSensX = cms.vdouble(),
      phiSensY = cms.vdouble(),
      resX = cms.vdouble(),
      norResX = cms.vdouble(),
      probX = cms.vdouble(),
      errXHit = cms.vdouble(),
      errXTrk = cms.vdouble(),
      errX = cms.vdouble(),
      errX2 = cms.vdouble(),   #squared error of residuals(X)
      
      #Additional parameters for Cuts on Pixel Hits (depending on track reconstruction)
      resY = cms.vdouble(),
      norResY = cms.vdouble(),
      probY = cms.vdouble(),
      errYHit = cms.vdouble(),
      errYTrk = cms.vdouble(),
      errY = cms.vdouble(),
      errY2 = cms.vdouble(),   #squared error of residuals(Y)
    ),
    
    #Define minimum number of selected hits for track selection (choose only tracks with enough good hits)
    minGoodHitsPerTrack = cms.uint32(0),
    
    #File containing TrackerTree with ideal Geometry
    TrackerTreeFile = cms.string(os.environ['CMSSW_BASE'] + '/src/Alignment/TrackerTreeGenerator/hists/TrackerTree.root'),
    
    #Sectors defining set of modules for common overview plots resp. APE values
    Sectors = cms.VPSet(),
    
    
    
    ## Tool 1: Switch on Analyzer mode with full set of overview plots
    analyzerMode = cms.bool(True),
    
    #Vary Histo's ranges for Overview Plots (for analyzer mode)
    zoomHists = cms.bool(True),
    
    #Special Filter for Residual Error Histograms, additional hists binned in 100um (1: 0-100um, 2: 100-200um), (for analyzer mode)
    vErrHists = cms.vuint32(),
    
    
    
    ## Tool 2: Switch on calculation of APE values
    calculateApe = cms.bool(False),
    
    #Define intervals in residual error for calculation of APE (one estimation per interval), (for APE calculation)
    residualErrorBinning = cms.vdouble(),
)

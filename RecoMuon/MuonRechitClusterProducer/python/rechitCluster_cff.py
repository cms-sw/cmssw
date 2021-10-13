import FWCore.ParameterSet.Config as cms

import RecoMuon.MuonRechitClusterProducer.CSCrechitClusterProducer_cfi as CSCcluster 
import RecoMuon.MuonRechitClusterProducer.DTrechitClusterProducer_cfi as DTcluster

ca4CSCrechitClusters= CSCcluster.CSCrechitClusterProducer.clone(
    recHitLabel = "csc2DRecHits",
    nRechitMin  = 50,
    rParam      = 0.4,
    nStationThres = 10, 
) 
ca4DTrechitClusters = DTcluster.DTrechitClusterProducer.clone(
    recHitLabel = "dt1DRecHits",
    nRechitMin  = 50,
    rParam      = 0.4,
    nStationThres = 10, 
) 


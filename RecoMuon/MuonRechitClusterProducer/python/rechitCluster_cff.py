import FWCore.ParameterSet.Config as cms

import RecoMuon.MuonRechitClusterProducer.cscRechitClusterProducer_cfi as CSCcluster 
import RecoMuon.MuonRechitClusterProducer.dtRechitClusterProducer_cfi as DTcluster

ca4CSCrechitClusters= CSCcluster.cscRechitClusterProducer.clone(
    recHitLabel = "csc2DRecHits",
    nRechitMin  = 50,
    rParam      = 0.4,
    nStationThres = 10, 
) 
ca4DTrechitClusters = DTcluster.dtRechitClusterProducer.clone(
    recHitLabel = "dt1DRecHits",
    nRechitMin  = 50,
    rParam      = 0.4,
    nStationThres = 10, 
) 


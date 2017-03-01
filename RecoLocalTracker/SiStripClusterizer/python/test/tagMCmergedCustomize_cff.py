#
# With this customization the ClusterizerRefinerTagMCmerged module will be substituted
# for the standard clusterizer.  If a cluster is matched to more than one simTrack
# its "merged" bit will be set, so that SiStripCluster::isMerged() will return true.
#
# If pileup is present, add the following line so that only in-time simTracks will
# be counted, and make sure that process.mix is on the path. 
# process.siStripClusters.ClusterRefiner.associateRecoTracks = cms.bool(False)
#

import FWCore.ParameterSet.Config as cms

def tagMCmerged(process):

  process.siStripClustersUntagged = process.siStripClusters.clone()
  stripClusIndex = process.striptrackerlocalreco.index(process.siStripClusters)                                                                   
  process.striptrackerlocalreco.remove(process.siStripClusters)
  del process.siStripClusters
  process.load('RecoLocalTracker.SiStripClusterizer.test.ClusterRefinerTagMCmerged_cfi')
  process.siStripClustersTagged = cms.Sequence(process.siStripClustersUntagged*process.siStripClusters)
  process.striptrackerlocalreco.insert(stripClusIndex,process.siStripClustersTagged)

# Override the chargePerCM cut in stripCPE and use cluster::isMerged() instead.
  process.StripCPEfromTrackAngleESProducer.parameters.maxChgOneMIP = cms.double(-6000.)                                                                   

  return(process)

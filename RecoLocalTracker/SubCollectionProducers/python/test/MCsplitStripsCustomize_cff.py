#
# With this customization the ClusterMCsplitStrips module will be substituted
# for the standard clusterizer.  If a cluster is matched to more than one simTrack
# it will be split into the corresponding true clusters.
#

import FWCore.ParameterSet.Config as cms

def splitMCmerged(process):

  process.siStripClustersUnsplit = process.siStripClusters.clone()
  stripClusIndex = process.striptrackerlocalreco.index(process.siStripClusters)                                                                   
  process.striptrackerlocalreco.remove(process.siStripClusters)
  del process.siStripClusters
  process.load('RecoLocalTracker.SubCollectionProducers.test.ClusterMCsplitStrips_cfi')
  process.siStripClustersMCsplit = cms.Sequence(process.siStripClustersUnsplit*process.siStripClusters)
  process.striptrackerlocalreco.insert(stripClusIndex,process.siStripClustersMCsplit)
 
# Override the chargePerCM cut in stripCPE
  process.StripCPEfromTrackAngleESProducer.parameters.maxChgOneMIP = cms.double(-6000.)

  return(process)

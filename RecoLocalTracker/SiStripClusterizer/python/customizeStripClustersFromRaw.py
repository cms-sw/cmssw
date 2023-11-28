import FWCore.ParameterSet.Config as cms

# replace the standard SiStripClusterizer with the switch producer
# meant primarily for testing
def customizeStripClustersFromRaw(process):
    if hasattr(process, 'striptrackerlocalrecoTask'):
        process.striptrackerlocalrecoTask.remove(process.siStripClusters)
        process.load("RecoLocalTracker.SiStripClusterizer.SiStripClusterizerOnDemand_cfi")
        # CPU should emulate the full detector clusterizer
        process.siStripClusters.cpu.onDemand = cms.bool(False)
        process.striptrackerlocalrecoTask.add(process.siStripClustersTask)

    return process

def customizeHLTStripClustersFromRaw(process):
    if hasattr(process, 'hltSiStripRawToClustersFacility'):
        import RecoLocalTracker.SiStripClusterizer.SiStripClusterizerOnDemand_cfi as SiStripClusterizerOnDemand_cfi

        process.load("RecoLocalTracker.SiStripClusterizer.SiStripClusterizerOnDemand_cfi")
        process.hltSiStripRawToClustersFacility = SiStripClusterizerOnDemand_cfi.siStripClusters.clone()
        process.HLTDoLocalStripSequence.replace(process.hltSiStripRawToClustersFacility,
                                   cms.Sequence(process.hltSiStripRawToClustersFacility, process.siStripClustersTaskCUDA))

    return process

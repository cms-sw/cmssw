import FWCore.ParameterSet.Config as cms

def custom_ntuples_layer1_truncation(process):
    ntuples = process.l1tHGCalTriggerNtuplizer.Ntuples
    for ntuple in ntuples:
        if ntuple.NtupleName=='HGCalTriggerNtupleHGCClusters' or \
           ntuple.NtupleName=='HGCalTriggerNtupleHGCTriggerCells' or \
           ntuple.NtupleName=='HGCalTriggerNtupleHGCMulticlusters':
            ntuple.Clusters = cms.InputTag('l1tHGCalBackEndLayer1Producer:HGCalBackendLayer1Processor')
    return process

def custom_ntuples_stage1_truncation(process):
    ntuples = process.l1tHGCalTriggerNtuplizer.Ntuples
    for ntuple in ntuples:
        if ntuple.NtupleName=='HGCalTriggerNtupleHGCClusters' or \
           ntuple.NtupleName=='HGCalTriggerNtupleHGCTriggerCells' or \
           ntuple.NtupleName=='HGCalTriggerNtupleHGCMulticlusters':
            ntuple.Clusters = cms.InputTag('l1tHGCalBackEndStage1Producer:HGCalBackendStage1Processor')
            ntuple.Multiclusters = cms.InputTag('l1tHGCalBackEndStage2Producer:HGCalBackendLayer2Processor3DClustering')
    return process

def custom_ntuples_standalone_clustering(process):
    ntuples = process.l1tHGCalTriggerNtuplizer.Ntuples
    for ntuple in ntuples:
        if ntuple.NtupleName=='HGCalTriggerNtupleHGCTriggerCells' or \
           ntuple.NtupleName=='HGCalTriggerNtupleHGCClusters' or \
           ntuple.NtupleName=='HGCalTriggerNtupleHGCMulticlusters':
            ntuple.Multiclusters = cms.InputTag('l1tHGCalBackEndLayer2Producer:HGCalBackendLayer2Processor3DClusteringSA')
    return process


def custom_ntuples_standalone_tower(process):
    ntuples = process.l1tHGCalTriggerNtuplizer.Ntuples
    for ntuple in ntuples:
        if ntuple.NtupleName=='HGCalTriggerNtupleHGCTowers':
            ntuple.Towers = cms.InputTag('l1tHGCalTowerProducer:HGCalTowerProcessorSA')
    return process


class CreateNtuple(object):
    def __init__(self,
        ntuple_list=[
            'event',
            'gen', 'genjet', 'gentau',
            'digis',
            'triggercells',
            'clusters', 'multiclusters'
            ]
            ):
        self.ntuple_list = ntuple_list

    def __call__(self, process, inputs):
        vpset = []
        for ntuple in self.ntuple_list:
            pset = getattr(process, 'ntuple_'+ntuple).clone()
            if ntuple=='triggercells':
                pset.TriggerCells = cms.InputTag(inputs[0])
                pset.Multiclusters = cms.InputTag(inputs[2])
            elif ntuple=='clusters':
                pset.Clusters = cms.InputTag(inputs[1])
                pset.Multiclusters = cms.InputTag(inputs[2])
            elif ntuple=='multiclusters':
                pset.Multiclusters = cms.InputTag(inputs[2])
            vpset.append(pset)
        ntuplizer = process.l1tHGCalTriggerNtuplizer.clone()
        ntuplizer.Ntuples = cms.VPSet(vpset)
        return ntuplizer

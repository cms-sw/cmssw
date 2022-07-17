import FWCore.ParameterSet.Config as cms

def custom_ntuples_layer1_truncation(process):
    ntuples = process.hgcalTriggerNtuplizer.Ntuples
    for ntuple in ntuples:
        if ntuple.NtupleName=='HGCalTriggerNtupleHGCClusters' or \
           ntuple.NtupleName=='HGCalTriggerNtupleHGCTriggerCells' or \
           ntuple.NtupleName=='HGCalTriggerNtupleHGCMulticlusters':
            ntuple.Clusters = cms.InputTag('hgcalBackEndLayer1Producer:HGCalBackendLayer1Processor')
    return process

def custom_ntuples_stage1_truncation(process):
    ntuples = process.hgcalTriggerNtuplizer.Ntuples
    for ntuple in ntuples:
        if ntuple.NtupleName=='HGCalTriggerNtupleHGCClusters' or \
           ntuple.NtupleName=='HGCalTriggerNtupleHGCTriggerCells' or \
           ntuple.NtupleName=='HGCalTriggerNtupleHGCMulticlusters':
            ntuple.Clusters = cms.InputTag('hgcalBackEndStage1Producer:HGCalBackendStage1Processor')
            ntuple.Multiclusters = cms.InputTag('hgcalBackEndStage2Producer:HGCalBackendLayer2Processor3DClustering')
    return process

def custom_ntuples_standalone_clustering(process):
    ntuples = process.hgcalTriggerNtuplizer.Ntuples
    for ntuple in ntuples:
        if ntuple.NtupleName=='HGCalTriggerNtupleHGCTriggerCells' or \
           ntuple.NtupleName=='HGCalTriggerNtupleHGCClusters' or \
           ntuple.NtupleName=='HGCalTriggerNtupleHGCMulticlusters':
            ntuple.Multiclusters = cms.InputTag('hgcalBackEndLayer2Producer:HGCalBackendLayer2Processor3DClusteringSA')
    return process


def custom_ntuples_standalone_tower(process):
    ntuples = process.hgcalTriggerNtuplizer.Ntuples
    for ntuple in ntuples:
        if ntuple.NtupleName=='HGCalTriggerNtupleHGCTowers':
            ntuple.Towers = cms.InputTag('hgcalTowerProducer:HGCalTowerProcessorSA')
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
        ntuplizer = process.hgcalTriggerNtuplizer.clone()
        ntuplizer.Ntuples = cms.VPSet(vpset)
        return ntuplizer

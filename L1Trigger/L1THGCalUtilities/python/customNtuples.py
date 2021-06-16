import FWCore.ParameterSet.Config as cms

def custom_ntuples_V9(process):
    ntuples = process.hgcalTriggerNtuplizer.Ntuples
    for ntuple in ntuples:
        if ntuple.NtupleName=='HGCalTriggerNtupleHGCDigis' or \
           ntuple.NtupleName=='HGCalTriggerNtupleHGCTriggerCells':
            ntuple.bhSimHits = cms.InputTag('g4SimHits:HGCHitsHEback')
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


def create_ntuple(process, inputs,
        ntuple_list=[
            'event',
            'gen', 'genjet', 'gentau',
            'digis',
            'triggercells',
            'clusters', 'multiclusters'
            ]
        ):
    vpset = []
    for ntuple in ntuple_list:
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





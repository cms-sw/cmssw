import FWCore.ParameterSet.Config as cms

def custom_ntuples_V9(process):
    ntuples = process.hgcalTriggerNtuplizer.Ntuples
    for ntuple in ntuples:
        if ntuple.NtupleName=='HGCalTriggerNtupleHGCDigis' or \
           ntuple.NtupleName=='HGCalTriggerNtupleHGCTriggerCells':
            ntuple.bhSimHits = cms.InputTag('g4SimHits:HGCHitsHEback')
    return process

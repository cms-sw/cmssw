import FWCore.ParameterSet.Config as cms

from L1Trigger.L1THGCalUtilities.hgcalTriggerNtuples_cfi import *

l1CaloTriggerNtuplizer = hgcalTriggerNtuplizer.clone()

ntuple_multiclusters_hmvdr = ntuple_multiclusters.clone()
ntuple_multiclusters_hmvdr.Prefix = cms.untracked.string('HMvDR')

l1CaloTriggerNtuplizer.Ntuples = cms.VPSet(ntuple_event,
                                           ntuple_gen,
                                           ntuple_triggercells,
                                           ntuple_multiclusters_hmvdr)

from L1Trigger.L1CaloTrigger.ntuple_cfi import *

l1CaloTriggerNtuplizer.Ntuples.append(ntuple_egammaEE)
l1CaloTriggerNtuplizer.Ntuples.append(ntuple_egammaEB)
l1CaloTriggerNtuplizer.Ntuples.append(ntuple_TTTracks)
l1CaloTriggerNtuplizer.Ntuples.append(ntuple_tkEleEllEE)
l1CaloTriggerNtuplizer.Ntuples.append(ntuple_tkEleEllEB)

l1CaloTriggerNtuples = cms.Sequence(l1CaloTriggerNtuplizer)


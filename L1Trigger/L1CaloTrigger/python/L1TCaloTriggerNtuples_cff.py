import FWCore.ParameterSet.Config as cms

from L1Trigger.L1THGCalUtilities.hgcalTriggerNtuples_cfi import *

l1CaloTriggerNtuplizer = hgcalTriggerNtuplizer.clone()

ntuple_multiclusters_hmvdr = ntuple_multiclusters.clone()
ntuple_multiclusters_hmvdr.Prefix = cms.untracked.string('HMvDR')

l1CaloTriggerNtuplizer.Ntuples = cms.VPSet(ntuple_event,
                                           ntuple_gen,
                                           ntuple_triggercells,
                                           ntuple_multiclusters_hmvdr)

# l1CaloTriggerNtuplizer.Ntuples.remove(ntuple_genjet)
# l1CaloTriggerNtuplizer.Ntuples.remove(ntuple_gentau)
# l1CaloTriggerNtuplizer.Ntuples.remove(ntuple_digis)
# l1CaloTriggerNtuplizer.Ntuples.remove(ntuple_multiclusters)
# l1CaloTriggerNtuplizer.Ntuples.remove(ntuple_towers)


from L1Trigger.L1CaloTrigger.ntuple_cfi import *





l1CaloTriggerNtuplizer.Ntuples.append(ntuple_egammaEE)
l1CaloTriggerNtuplizer.Ntuples.append(ntuple_egammaEB)
l1CaloTriggerNtuplizer.Ntuples.append(ntuple_TTTracks)
l1CaloTriggerNtuplizer.Ntuples.append(ntuple_tkEleEllEE)
l1CaloTriggerNtuplizer.Ntuples.append(ntuple_tkEleEllEB)

#

l1CaloTriggerNtuples = cms.Sequence(l1CaloTriggerNtuplizer)


# from Configuration.Eras.Modifier_phase2_hgcalV9_cff import phase2_hgcalV9
# from L1Trigger.L1THGCalUtilities.customNtuples import custom_ntuples_V9
# modifyHgcalTriggerNtuplesWithV9Geometry_ = phase2_hgcalV9.makeProcessModifier(custom_ntuples_V9)
#

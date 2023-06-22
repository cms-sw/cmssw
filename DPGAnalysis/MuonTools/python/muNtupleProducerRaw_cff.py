import FWCore.ParameterSet.Config as cms

from DPGAnalysis.MuonTools.muNtupleDTDigiFiller_cfi import muNtupleDTDigiFiller
from DPGAnalysis.MuonTools.muNtupleCSCALCTDigiFiller_cfi import muNtupleCSCALCTDigiFiller
from DPGAnalysis.MuonTools.muNtupleCSCWireDigiFiller_cfi import muNtupleCSCWireDigiFiller
from DPGAnalysis.MuonTools.muNtupleRPCDigiFiller_cfi import muNtupleRPCDigiFiller
from DPGAnalysis.MuonTools.muNtupleGEMDigiFiller_cfi import muNtupleGEMDigiFiller

from DPGAnalysis.MuonTools.muNtupleRPCRecHitFiller_cfi import muNtupleRPCRecHitFiller
from DPGAnalysis.MuonTools.muNtupleGEMRecHitFiller_cfi import muNtupleGEMRecHitFiller

muNtupleProducerRaw = cms.Sequence(muNtupleDTDigiFiller 
                                   + muNtupleRPCDigiFiller 
                                   + muNtupleGEMDigiFiller 
                                   + muNtupleRPCRecHitFiller 
                                   + muNtupleGEMRecHitFiller
                                   + muNtupleCSCALCTDigiFiller
                                   + muNtupleCSCWireDigiFiller
                                  )
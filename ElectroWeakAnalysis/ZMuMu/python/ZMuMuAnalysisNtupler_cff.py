import FWCore.ParameterSet.Config as cms

from ElectroWeakAnalysis.ZMuMu.ZMuMuCategoriesSequences_cff import *

import copy



goodZToMuMuEdmNtuple = cms.EDProducer(
    "CandViewNtpProducer", 
    src=cms.InputTag("goodZToMuMuAtLeast1HLTLoose"),
    lazyParser=cms.untracked.bool(True),
    prefix=cms.untracked.string("z"),
    eventInfo=cms.untracked.bool(True),
    variables = cms.VPSet(
    cms.PSet(
    tag = cms.untracked.string("Mass"),
    quantity = cms.untracked.string("mass")
    ),
    cms.PSet(
    tag = cms.untracked.string("Pt"),
    quantity = cms.untracked.string("pt")
    ),
    cms.PSet(
    tag = cms.untracked.string("Eta"),
    quantity = cms.untracked.string("eta")
    ),
    cms.PSet(
    tag = cms.untracked.string("Phi"),
    quantity = cms.untracked.string("phi")
    ),
    cms.PSet(
    tag = cms.untracked.string("Y"),
    quantity = cms.untracked.string("rapidity")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau1Pt"),
    quantity = cms.untracked.string("daughter(0).masterClone.pt")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau2Pt"),
    quantity = cms.untracked.string("daughter(1).masterClone.pt")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau1Q"),
    quantity = cms.untracked.string("daughter(0).masterClone.charge")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau2Q"),
    quantity = cms.untracked.string("daughter(1).masterClone.charge")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau1Eta"),
    quantity = cms.untracked.string("daughter(0).masterClone.eta")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau2Eta"),
    quantity = cms.untracked.string("daughter(1).masterClone.eta")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau1Phi"),
    quantity = cms.untracked.string("daughter(0).masterClone.phi")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau2Phi"),
    quantity = cms.untracked.string("daughter(1).masterClone.phi")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau1NofHit"),
    quantity = cms.untracked.string("daughter(0).masterClone.numberOfValidHits")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau1NofHitTk"),
    quantity = cms.untracked.string("daughter(0).masterClone.innerTrack.numberOfValidHits")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau1NofHitSta"),
    quantity = cms.untracked.string("daughter(0).masterClone.outerTrack.numberOfValidHits")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau1NofMuChambers"),
    quantity = cms.untracked.string("daughter(0).masterClone.numberOfChambers")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau1NofMuMatches"),
    quantity = cms.untracked.string("daughter(0).masterClone.numberOfMatches")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau1Chi2"),
    quantity = cms.untracked.string("daughter(0).masterClone.normChi2")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau1dB"),
    quantity = cms.untracked.string("daughter(0).masterClone.dB")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau1Iso"),
    quantity = cms.untracked.string("daughter(0).masterClone.userIso(0)")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau2Iso"),
    quantity = cms.untracked.string("daughter(1).masterClone.userIso(0)")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau1relIso"),
    quantity = cms.untracked.string("daughter(0).masterClone.userIso(1)")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau2relIso"),
    quantity = cms.untracked.string("daughter(1).masterClone.userIso(1)")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau1TrkIso"),
    quantity = cms.untracked.string("daughter(0).masterClone.trackIso")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau2TrkIso"),
    quantity = cms.untracked.string("daughter(1).masterClone.trackIso")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau1EcalIso"),
    quantity = cms.untracked.string("daughter(0).masterClone.ecalIso")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau2EcalIso"),
    quantity = cms.untracked.string("daughter(1).masterClone.ecalIso")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau1HcalIso"),
    quantity = cms.untracked.string("daughter(0).masterClone.hcalIso")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau2HcalIso"),
    quantity = cms.untracked.string("daughter(1).masterClone.hcalIso")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau1dxyFromBS"),
    quantity = cms.untracked.string("daughter(0).masterClone.userFloat('zDau_dxyFromBS')")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau1dzFromBS"),
    quantity = cms.untracked.string("daughter(0).masterClone.userFloat('zDau_dzFromBS')")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau1dxyFromPV"),
    quantity = cms.untracked.string("daughter(0).masterClone.userFloat('zDau_dxyFromPV')")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau1dzFromPV"),
    quantity = cms.untracked.string("daughter(0).masterClone.userFloat('zDau_dzFromPV')")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau1HLTBit"),
    quantity = cms.untracked.string("daughter(0).masterClone.userFloat('zDau_HLTBit')")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau2HLTBit"),
    quantity = cms.untracked.string("daughter(1).masterClone.userFloat('zDau_HLTBit')")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau2dxyFromBS"),
    quantity = cms.untracked.string("daughter(1).masterClone.userFloat('zDau_dxyFromBS')")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau2dzFromBS"),
    quantity = cms.untracked.string("daughter(1).masterClone.userFloat('zDau_dzFromBS')")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau2dxyFromPV"),
    quantity = cms.untracked.string("daughter(1).masterClone.userFloat('zDau_dxyFromPV')")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau2dzFromPV"),
    quantity = cms.untracked.string("daughter(1).masterClone.userFloat('zDau_dzFromPV')")
    ),
    cms.PSet(
    tag = cms.untracked.string("VtxNormChi2"),
    quantity = cms.untracked.string("vertexNormalizedChi2")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau1TrkChi2"),
    quantity = cms.untracked.string("daughter(0).masterClone.innerTrack.normalizedChi2")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau1MuEnergyHad"),
    quantity = cms.untracked.string("daughter(0).masterClone.calEnergy.had")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau1MuEnergyEm"),
    quantity = cms.untracked.string("daughter(0).masterClone.calEnergy.em")
    ),
    cms.PSet(
    tag = cms.untracked.string("TrueMass"),
    quantity = cms.untracked.string("userFloat('TrueMass')")
    ),
    cms.PSet(
    tag = cms.untracked.string("TruePt"),
    quantity = cms.untracked.string("userFloat('TruePt')")
    ),   
    cms.PSet(
    tag = cms.untracked.string("TrueEta"),
    quantity = cms.untracked.string("userFloat('TrueEta')")
    ),
    cms.PSet(
    tag = cms.untracked.string("TruePhi"),
    quantity = cms.untracked.string("userFloat('TruePhi')")
    ),
    cms.PSet(
    tag = cms.untracked.string("TrueY"),
    quantity = cms.untracked.string("userFloat('TrueY')")
    )
   
    )
    
    )


zGolden=(
    cms.PSet(
    tag = cms.untracked.string("Dau2NofHit"),
    quantity = cms.untracked.string("daughter(1).masterClone.numberOfValidHits")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau2NofHitTk"),
    quantity = cms.untracked.string("daughter(1).masterClone.innerTrack.numberOfValidHits")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau2NofHitSta"),
    quantity = cms.untracked.string("daughter(1).masterClone.outerTrack.numberOfValidHits")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau2NofMuChambers"),
    quantity = cms.untracked.string("daughter(1).masterClone.numberOfChambers")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau2NofMuMatches"),
    quantity = cms.untracked.string("daughter(1).masterClone.numberOfMatches")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau2Chi2"),
    quantity = cms.untracked.string("daughter(1).masterClone.normChi2")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau2dB"),
    quantity = cms.untracked.string("daughter(1).masterClone.dB")
    ),
    cms.PSet(
    tag = cms.untracked.string("MassSa"),
    quantity = cms.untracked.string("userFloat('MassSa')")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau1SaPt"),
    quantity = cms.untracked.string("userFloat('Dau1SaPt')")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau2SaPt"),
    quantity = cms.untracked.string("userFloat('Dau1SaPt')")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau1SaEta"),
    quantity = cms.untracked.string("userFloat('Dau1SaEta')")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau2SaEta"),
    quantity = cms.untracked.string("userFloat('Dau2SaEta')")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau1SaPhi"),
    quantity = cms.untracked.string("userFloat('Dau1SaPhi')")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau2SaPhi"),
    quantity = cms.untracked.string("userFloat('Dau2SaPhi')")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau2TrkChi2"),
    quantity = cms.untracked.string("daughter(1).masterClone.innerTrack.normalizedChi2")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau2MuEnergyHad"),
    quantity = cms.untracked.string("daughter(1).masterClone.calEnergy.had")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau2MuEnergyEm"),
    quantity = cms.untracked.string("daughter(1).masterClone.calEnergy.em")
    )
    
                
    )

zMuTrk=(
    cms.PSet(
    tag = cms.untracked.string("Dau2NofHit"),
    quantity = cms.untracked.string("daughter(1).masterClone.track.numberOfValidHits")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau2NofHitTk"),
    quantity = cms.untracked.string("daughter(1).masterClone.track.numberOfValidHits")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau2Chi2"),
    quantity = cms.untracked.string("daughter(1).masterClone.track.normalizedChi2")
    )
            
    )



goodZToMuMuEdmNtupleLoose= copy.deepcopy(goodZToMuMuEdmNtuple)
goodZToMuMuEdmNtupleLoose.variables += zGolden
goodZToMuMuEdmNtupleLoose.prefix = cms.untracked.string("zGolden")
goodZToMuMuPathLoose.__iadd__(goodZToMuMuEdmNtupleLoose)
goodZToMuMuPathLoose.setLabel("goodZToMuMuEdmLoose")


goodZToMuMuSameChargeEdmNtupleLoose= copy.deepcopy(goodZToMuMuEdmNtuple)
goodZToMuMuSameChargeEdmNtupleLoose.src = cms.InputTag("goodZToMuMuSameChargeAtLeast1HLTLoose")
goodZToMuMuSameChargeEdmNtupleLoose.prefix = cms.untracked.string("zSameCharge")
goodZToMuMuSameChargeEdmNtupleLoose.variables += zGolden
goodZToMuMuSameChargePathLoose.__iadd__(goodZToMuMuSameChargeEdmNtupleLoose)
goodZToMuMuSameChargePathLoose.setLabel("goodZToMuMuSameChargeLoose")



goodZToMuMuOneStandAloneEdmNtupleLoose= copy.deepcopy(goodZToMuMuEdmNtuple)
goodZToMuMuOneStandAloneEdmNtupleLoose.src=cms.InputTag("goodZToMuMuOneStandAloneMuonFirstHLTLoose")
goodZToMuMuOneStandAloneEdmNtupleLoose.prefix=cms.untracked.string("zMuSta")
goodZToMuMuOneStandAloneEdmNtupleLoose.variables += zGolden
goodZToMuMuOneStandAloneMuonPathLoose.__iadd__(goodZToMuMuOneStandAloneEdmNtupleLoose)
goodZToMuMuOneStandAloneMuonPathLoose.setLabel("goodZToMuMuOneStandAloneMuonLoose")

#goodZToMuMuOneTrackEdmNtupleLoose= copy.deepcopy(goodZToMuMuEdmNtuple)
#goodZToMuMuOneTrackEdmNtupleLoose.src=cms.InputTag("goodZToMuMuOneTrackFirstHLTLoose")
#goodZToMuMuOneTrackEdmNtupleLoose.prefix=cms.untracked.string("zMuTrk")
#goodZToMuMuOneTrackEdmNtupleLoose.variables += zMuTrk
#goodZToMuMuOneTrackPathLoose.__iadd__(goodZToMuMuOneTrackEdmNtupleLoose)
#goodZToMuMuOneTrackPathLoose.setLabel("goodZToMuMuOneTrackLoose")

ntuplesOut = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string('NtupleLooseTestNew.root'),
    outputCommands = cms.untracked.vstring(
      "drop *",
      "keep *_goodZToMuMuEdmNtupleLoose_*_*",
      "keep *_goodZToMuMuSameChargeEdmNtupleLoose_*_*",
      "keep *_goodZToMuMuOneStandAloneEdmNtupleLoose_*_*",
      #"keep *_goodZToMuMuOneTrackEdmNtupleLoose_*_*"
      
    )
    )


ntuplesOut.setLabel("ntuplesOut")
NtuplesOut.__iadd__(ntuplesOut)
NtuplesOut.setLabel("NtuplesOut")



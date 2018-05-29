import FWCore.ParameterSet.Config as cms

reducedHcalRecHits = cms.EDProducer("HcalHitSelection",
                                    hbheTag = cms.InputTag('hbhereco'),
                                    hfTag = cms.InputTag('hfreco'),
                                    hoTag = cms.InputTag('horeco'),
                                    hoSeverityLevel = cms.int32(13),
                                    interestingDetIds = cms.VInputTag(
                                         cms.InputTag("interestingGedEgammaIsoHCALDetId"),
                                         cms.InputTag("interestingOotEgammaIsoHCALDetId"),
                                         )
                                    )

from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
_pp_on_AA_interestingDetIds  = cms.VInputTag([cms.InputTag("interestingGedEgammaIsoHCALDetId")])

pp_on_AA_2018.toModify(
    reducedHcalRecHits,
    interestingDetIds = _pp_on_AA_interestingDetIds
    )

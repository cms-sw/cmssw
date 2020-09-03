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

slimmedHcalRecHits = reducedHcalRecHits.clone(
          hbheTag = cms.InputTag("reducedHcalRecHits","hbhereco"),
          hfTag   = cms.InputTag("reducedHcalRecHits","hfreco"),
          hoTag   = cms.InputTag(""),
          interestingDetIds = cms.VInputTag()
       )

from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
pp_on_AA_2018.toModify(reducedHcalRecHits.interestingDetIds, func = lambda list: list.remove(cms.InputTag("interestingOotEgammaIsoHCALDetId")) )

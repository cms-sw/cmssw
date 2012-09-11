import FWCore.ParameterSet.Config as cms

CaloMode = 1
# 0: custom local reco bypassing digis, ECAL and HCAL
# 1: as 0, but full digi + std local reco in ECAL
# 2: as 0, but full digi + std local reco in HCAL
# 3: full digi + std local reco in ECAL and HCAL

if(CaloMode==0):

    from FastSimulation.CaloRecHitsProducer.EcalRecHitsCustom_cff import *
    from FastSimulation.CaloRecHitsProducer.HcalRecHitsCustom_cff import *

    caloRecHits = cms.Sequence(ecalRecHit*ecalPreshowerRecHit*hbhereco*horeco*hfreco)

if(CaloMode==1):

    from FastSimulation.CaloRecHitsProducer.EcalDigisPlusRecHits_cff import *
    from FastSimulation.CaloRecHitsProducer.HcalRecHitsCustom_cff import *

    caloDigis = cms.Sequence(ecalDigisSequence)
    caloRecHits = cms.Sequence(ecalRecHitSequence*ecalPreshowerRecHit*hbhereco*horeco*hfreco)
    
if(CaloMode==2):

    from FastSimulation.CaloRecHitsProducer.EcalRecHitsCustom_cff import *
    from FastSimulation.CaloRecHitsProducer.HcalDigisPlusRecHits_cff import *

    caloDigis = cms.Sequence(hcalDigisSequence)
    caloRecHits = cms.Sequence(ecalRecHit*ecalPreshowerRecHit*hcalRecHitSequence)

if(CaloMode==3):

    from FastSimulation.CaloRecHitsProducer.EcalDigisPlusRecHits_cff import *
    from FastSimulation.CaloRecHitsProducer.HcalDigisPlusRecHits_cff import *

    caloDigis = cms.Sequence(ecalDigisSequence*hcalDigisSequence)
    caloRecHits = cms.Sequence(ecalRecHitSequence*ecalPreshowerRecHit*hcalRecHitSequence)


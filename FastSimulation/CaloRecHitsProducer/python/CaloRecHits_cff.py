import FWCore.ParameterSet.Config as cms

from FastSimulation.Configuration.CommonInputs_cff import *

#CaloMode is defined in CommonInputs
# 0: custom local reco bypassing digis, ECAL and HCAL - it was the only option available until 60X
# 1: as 0, but full digi + std local reco in ECAL - default since 610pre6
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
    caloRecHits = cms.Sequence(ecalRecHitSequence*hbhereco*horeco*hfreco)
    
if(CaloMode==2):
  
    from FastSimulation.CaloRecHitsProducer.EcalRecHitsCustom_cff import *
    from FastSimulation.CaloRecHitsProducer.HcalDigisPlusRecHits_cff import *

    caloDigis = cms.Sequence(hcalDigisSequence)
    caloRecHits = cms.Sequence(ecalRecHit*ecalPreshowerRecHit*hcalRecHitSequence)

if(CaloMode==3):

    from FastSimulation.CaloRecHitsProducer.FullDigisPlusRecHits_cff import *
    caloDigis = cms.Sequence(DigiSequence)
    caloRecHits = cms.Sequence(ecalRecHitSequence*hcalRecHitSequence)
    


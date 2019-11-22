import FWCore.ParameterSet.Config as cms

#
# Ecal part
#
from RecoLocalCalo.Configuration.ecalLocalRecoSequence_cff import *
from RecoLocalCalo.EcalRecAlgos.EcalSeverityLevelESProducer_cfi import *

#defines a sequence ecalLocalRecoSequence
#
# Hcal part
#
# calo geometry
#
# changed by tommaso. now the calibrations are read from Configuration/StaqndardSequences/data/*Conditions.cff
#
#HCAL reconstruction
from RecoLocalCalo.Configuration.hcalLocalReco_cff import *
from RecoLocalCalo.Configuration.hcalGlobalReco_cff import *
#
# sequence CaloLocalReco and CaloGlobalReco
#
calolocalrecoTask = cms.Task(ecalLocalRecoTask,hcalLocalRecoTask)
calolocalreco = cms.Sequence(calolocalrecoTask)

from RecoLocalCalo.HcalRecProducers.HcalHitSelection_cfi import *
reducedHcalRecHitsTask = cms.Task( reducedHcalRecHits )
reducedHcalRecHitsSequence = cms.Sequence(reducedHcalRecHitsTask)
#
# R.Ofierzynski (29.Oct.2009): add NZS sequence
#
from RecoLocalCalo.Configuration.hcalLocalRecoNZS_cff import *
calolocalrecoTaskNZS = cms.Task(ecalLocalRecoTask,hcalLocalRecoTask,hcalLocalRecoTaskNZS)
calolocalrecoNZS = cms.Sequence(calolocalrecoTaskNZS)

from RecoLocalCalo.Configuration.hgcalLocalReco_cff import *
_phase2_calolocalrecoTask = calolocalrecoTask.copy()
_phase2_calolocalrecoTask.add(hgcalLocalRecoTask)

from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
phase2_hgcal.toReplaceWith( calolocalrecoTask , _phase2_calolocalrecoTask )

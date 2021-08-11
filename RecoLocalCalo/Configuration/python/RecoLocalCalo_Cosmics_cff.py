import FWCore.ParameterSet.Config as cms

#
# Ecal part
#
from RecoLocalCalo.Configuration.ecalLocalRecoSequenceCosmics_cff import *
from RecoLocalCalo.EcalRecAlgos.EcalSeverityLevelESProducer_cfi import *

#defines a sequence ecalLocalRecoSequence

#
# Hcal part
#
#HCAL reconstruction
import RecoLocalCalo.Configuration.hcalLocalReco_cff as _hcalLocalReco_cff
from RecoLocalCalo.HcalRecAlgos.hcalRecAlgoESProd_cfi import *
from RecoLocalCalo.HcalRecAlgos.hcalChannelPropertiesESProd_cfi import *
#
# sequence CaloLocalReco
#

hbhereco = _hcalLocalReco_cff.hbheprereco.cpu.clone(
    tsFromDB = False,
    recoParamsFromDB = False,
    algorithm = dict(
        useMahi = False,
        useM2 = False,
        useM3 = False,
        firstSampleShift = -1000,
        samplesToAdd = 10,
        correctForPhaseContainment = False,
    ),
    sipmQTSShift = -100,
    sipmQNTStoSum = 200,
)
hfreco = _hcalLocalReco_cff._default_hfreco.clone(
    firstSample = 0,
    samplesToAdd = 10, ### min(10,size) in the algo
    correctForTimeslew = False,
    correctForPhaseContainment = False,
    tsFromDB = False,
    recoParamsFromDB = False,
    digiTimeFromDB = False,
)
horeco = _hcalLocalReco_cff.horeco.clone(
    firstSample = 0,
    samplesToAdd = 10,
    correctForTimeslew = False,
    correctForPhaseContainment = False,
    tsFromDB = False,
    recoParamsFromDB = False,
)
zdcreco = _hcalLocalReco_cff.zdcreco.clone(
#    firstSample = 1,
#    samplesToAdd = 8,
    correctForTimeslew = True,
    correctForPhaseContainment = True,
    correctionPhaseNS = 10.,
)

# 2017 customs
from Configuration.Eras.Modifier_run2_HF_2017_cff import run2_HF_2017

_phase1_hfreco = _hcalLocalReco_cff._phase1_hfreco.clone(
    algorithm = dict(
        Class = "HFSimpleTimeCheck",
        rejectAllFailures = False,
    )
)


run2_HF_2017.toReplaceWith(hfreco, _phase1_hfreco )

hfprereco = _hcalLocalReco_cff.hfprereco.clone(
    sumAllTimeSlices = True
)

from RecoLocalCalo.HcalRecProducers.hbheplan1_cfi import hbheplan1

# redefine hcal sequence
hcalLocalRecoTask = cms.Task(hbhereco,hfreco,horeco,zdcreco)
hcalLocalRecoSequence = cms.Sequence(hcalLocalRecoTask)

_phase1_hcalLocalRecoTask = hcalLocalRecoTask.copy()
_phase1_hcalLocalRecoTask.add(hfprereco)
run2_HF_2017.toReplaceWith(hcalLocalRecoTask, _phase1_hcalLocalRecoTask)

# shuffle modules so "hbheplan1" produces final collection of hits named "hbhereco"
_plan1_hcalLocalRecoTask = _phase1_hcalLocalRecoTask.copy()
hbheprereco = hbhereco.clone()
_plan1_hcalLocalRecoTask.add(hbheprereco)
from Configuration.Eras.Modifier_run2_HEPlan1_2017_cff import run2_HEPlan1_2017
run2_HEPlan1_2017.toReplaceWith(hbhereco, hbheplan1)
run2_HEPlan1_2017.toReplaceWith(hcalLocalRecoTask, _plan1_hcalLocalRecoTask)

hbhecollapse = hbheplan1.clone()
_collapse_hcalLocalRecoTask = _phase1_hcalLocalRecoTask.copy()
_collapse_hcalLocalRecoTask.add(hbheprereco)
from Configuration.ProcessModifiers.run2_HECollapse_2018_cff import run2_HECollapse_2018
run2_HECollapse_2018.toReplaceWith(hbhereco, hbhecollapse)
run2_HECollapse_2018.toReplaceWith(hcalLocalRecoTask, _collapse_hcalLocalRecoTask)
calolocalrecoTaskCosmics = cms.Task(ecalLocalRecoTaskCosmics,hcalLocalRecoTask)
calolocalrecoCosmics = cms.Sequence(calolocalrecoTaskCosmics)
#
# R.Ofierzynski (29.Oct.2009): add NZS sequence
#
from RecoLocalCalo.Configuration.hcalLocalRecoNZS_cff import *
calolocalrecoTaskCosmicsNZS = cms.Task(ecalLocalRecoTaskCosmics,hcalLocalRecoTask,hcalLocalRecoTaskNZS) 
calolocalrecoCosmicsNZS = cms.Sequence(calolocalrecoTaskCosmicsNZS) 

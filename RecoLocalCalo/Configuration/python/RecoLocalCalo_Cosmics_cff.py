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
#
# sequence CaloLocalReco
#
hbheprereco = _hcalLocalReco_cff._default_hbheprereco.clone(
    puCorrMethod = 0,
    firstSample = 0,
    samplesToAdd = 10,
    correctForTimeslew = False,
    correctForPhaseContainment = False,
    tsFromDB = False,
    recoParamsFromDB = cms.bool(False),
)
hfreco = _hcalLocalReco_cff._default_hfreco.clone(
    firstSample = 0,
    samplesToAdd = 10, ### min(10,size) in the algo
    correctForTimeslew = False,
    correctForPhaseContainment = False,
    tsFromDB = False,
    recoParamsFromDB = cms.bool(False),
    digiTimeFromDB = False,
)
horeco = _hcalLocalReco_cff.horeco.clone(
    firstSample = 0,
    samplesToAdd = 10,
    correctForTimeslew = False,
    correctForPhaseContainment = False,
    tsFromDB = False,
    recoParamsFromDB = cms.bool(False),
)
zdcreco = _hcalLocalReco_cff.zdcreco.clone(
#    firstSample = 1,
#    samplesToAdd = 8,
    correctForTimeslew = True,
    correctForPhaseContainment = True,
    correctionPhaseNS = 10.,
)

# 2017 customs
from Configuration.Eras.Modifier_run2_HE_2017_cff import run2_HE_2017
from Configuration.Eras.Modifier_run2_HF_2017_cff import run2_HF_2017

_phase1_hbheprereco = _hcalLocalReco_cff._phase1_hbheprereco.clone(
    tsFromDB = cms.bool(False),
    recoParamsFromDB = cms.bool(False),
    algorithm = dict(
        useM2 = cms.bool(False),
        useM3 = cms.bool(False),
        firstSampleShift = cms.int32(-1000),
        samplesToAdd = cms.int32(10),
        correctForPhaseContainment = cms.bool(False),
    )
)

_phase1_hfreco = _hcalLocalReco_cff._phase1_hfreco.clone(
    algorithm = dict(
        Class = cms.string("HFSimpleTimeCheck"),
        rejectAllFailures = cms.bool(False),
    )
)


run2_HE_2017.toReplaceWith(hbheprereco, _phase1_hbheprereco )
run2_HF_2017.toReplaceWith(hfreco, _phase1_hfreco )

hfprereco = _hcalLocalReco_cff.hfprereco.clone(
    sumAllTimeSlices = cms.bool(True)
)

# redefine hcal sequence
hcalLocalRecoSequence = cms.Sequence(hbheprereco+hfreco+horeco+zdcreco)

_phase1_hcalLocalRecoSequence = hcalLocalRecoSequence.copy()
_phase1_hcalLocalRecoSequence.insert(0,hfprereco)
run2_HF_2017.toReplaceWith(hcalLocalRecoSequence, _phase1_hcalLocalRecoSequence)

calolocalrecoCosmics = cms.Sequence(ecalLocalRecoSequenceCosmics+hcalLocalRecoSequence)

#
# R.Ofierzynski (29.Oct.2009): add NZS sequence
#
from RecoLocalCalo.Configuration.hcalLocalRecoNZS_cff import *
calolocalrecoCosmicsNZS = cms.Sequence(ecalLocalRecoSequenceCosmics+hcalLocalRecoSequence+hcalLocalRecoSequenceNZS) 

import FWCore.ParameterSet.Config as cms

from CalibCalorimetry.CaloTPG.tpScales_cff import tpScales

CaloTPGTranscoder = cms.ESProducer("CaloTPGTranscoderULUTs",
    linearLUTs = cms.bool(False),
    nominal_gain = cms.double(0.177),
    tpScales = tpScales
)

from Configuration.Eras.Modifier_run2_HCAL_2018_cff import run2_HCAL_2018
run2_HCAL_2018.toModify(CaloTPGTranscoder, linearLUTs=True)

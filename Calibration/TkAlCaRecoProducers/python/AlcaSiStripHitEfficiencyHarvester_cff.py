import FWCore.ParameterSet.Config as cms

from Calibration.TkAlCaRecoProducers.AlcaSiStripHitEfficiencyHarvester_cfi import *
from DQMServices.Components.EDMtoMEConverter_cfi import *

EDMtoMEConvertSiStripHitEfficiency = EDMtoMEConverter.clone(
    lumiInputTag = ("MEtoEDMConvertSiStripHitEff","MEtoEDMConverterLumi"),
    runInputTag  = ("MEtoEDMConvertSiStripHitEff","MEtoEDMConverterRun")
)

DQMStore = cms.Service("DQMStore")
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
dqmEnvSiStripHitEfficiency = DQMEDHarvester('DQMHarvestingMetadata',
                                           subSystemFolder = cms.untracked.string('AlCaReco'))

ALCAHARVESTSiStripHitEfficiency = cms.Sequence(EDMtoMEConvertSiStripHitEfficiency + alcasiStripHitEfficiencyHarvester + dqmEnvSiStripHitEfficiency)

import FWCore.ParameterSet.Config as cms

from Calibration.TkAlCaRecoProducers.AlcaSiPixelLorentzAngleHarvesterMCS_cfi import *
from DQMServices.Components.EDMtoMEConverter_cfi import *

EDMtoMEConvertSiPixelLorentzAngleMCS = EDMtoMEConverter.clone(
    lumiInputTag = ("MEtoEDMConvertSiPixelLorentzAngleMCS","MEtoEDMConverterLumi"),
    runInputTag = ("MEtoEDMConvertSiPixelLorentzAngleMCS","MEtoEDMConverterRun")
)

DQMStore = cms.Service("DQMStore")

from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
dqmEnvSiPixelLorentzAngleMCS = DQMEDHarvester('DQMHarvestingMetadata',
                                              subSystemFolder = cms.untracked.string('AlCaReco'),  
                                           )

ALCAHARVESTSiPixelLorentzAngleMCS = cms.Sequence( EDMtoMEConvertSiPixelLorentzAngleMCS + alcaSiPixelLorentzAngleHarvesterMCS + dqmEnvSiPixelLorentzAngleMCS )

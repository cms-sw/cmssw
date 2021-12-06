import FWCore.ParameterSet.Config as cms

from Calibration.TkAlCaRecoProducers.AlcaSiPixelLorentzAngleHarvester_cfi import *
from DQMServices.Components.EDMtoMEConverter_cfi import *

EDMtoMEConvertSiPixelLorentzAngle = EDMtoMEConverter.clone(
    lumiInputTag = ("MEtoEDMConvertSiPixelLorentzAngle","MEtoEDMConverterLumi"),
    runInputTag = ("MEtoEDMConvertSiPixelLorentzAngle","MEtoEDMConverterRun")
)

DQMStore = cms.Service("DQMStore")

from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
dqmEnvSiPixelLorentzAngle = DQMEDHarvester('DQMHarvestingMetadata',
                                           subSystemFolder = cms.untracked.string('AlCaReco'),  
                                           )

ALCAHARVESTSiPixelLorentzAngle = cms.Sequence( EDMtoMEConvertSiPixelLorentzAngle + alcaSiPixelLorentzAngleHarvester + dqmEnvSiPixelLorentzAngle )

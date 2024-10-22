import FWCore.ParameterSet.Config as cms

from Calibration.TkAlCaRecoProducers.AlcaSiStripLorentzAngleHarvester_cfi import *
from DQMServices.Components.EDMtoMEConverter_cfi import *

EDMtoMEConvertSiStripLorentzAngle = EDMtoMEConverter.clone(
    lumiInputTag = ("MEtoEDMConvertSiStripLorentzAngle","MEtoEDMConverterLumi"),
    runInputTag = ("MEtoEDMConvertSiStripLorentzAngle","MEtoEDMConverterRun")
)

DQMStore = cms.Service("DQMStore")

from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
dqmEnvSiStripLorentzAngle = DQMEDHarvester('DQMHarvestingMetadata',
                                           subSystemFolder = cms.untracked.string('AlCaReco'),  
                                           )

ALCAHARVESTSiStripLorentzAngle = cms.Sequence( EDMtoMEConvertSiStripLorentzAngle + alcaSiStripLorentzAngleHarvester + dqmEnvSiStripLorentzAngle )

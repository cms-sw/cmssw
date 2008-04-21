import FWCore.ParameterSet.Config as cms

from HLTrigger.Egamma.EgammaHLTLocal_cff import *
from HLTrigger.Egamma.SingleElectronL1Isolated_cff import *
from HLTrigger.Egamma.SingleElectronL1NonIsolated_cff import *
from HLTrigger.Egamma.DoubleElectronL1Isolated_cff import *
from HLTrigger.Egamma.DoubleElectronL1NonIsolated_cff import *
from HLTrigger.Egamma.SinglePhotonL1Isolated_cff import *
from HLTrigger.Egamma.SinglePhotonL1NonIsolated_cff import *
from HLTrigger.Egamma.DoublePhotonL1Isolated_cff import *
from HLTrigger.Egamma.DoublePhotonL1NonIsolated_cff import *
from HLTrigger.Egamma.SingleEMHighEtL1NonIsolated_cff import *
from HLTrigger.Egamma.SingleEMVeryHighEtL1NonIsolated_cff import *
from HLTrigger.Egamma.ZeeCounter_cff import *
from HLTrigger.Egamma.DoubleExclusiveElectronL1Isolated_cff import *
from HLTrigger.Egamma.DoubleExclusivePhotonL1Isolated_cff import *
from HLTrigger.Egamma.SinglePhotonL1IsolatedPrescaledEt12_1e32_cff import *
from HLTrigger.Egamma.SingleElectronL1IsolatedLargeWindow_1e32_cff import *
from HLTrigger.Egamma.SingleElectronL1NonIsolatedLargeWindow_1e32_cff import *
from HLTrigger.Egamma.DoubleElectronL1IsolatedLargeWindow_1e32_cff import *
from HLTrigger.Egamma.DoubleElectronL1NonIsolatedLargeWindow_1e32_cff import *
HLT1Electron = cms.Path(singleElectronL1Isolated+cms.SequencePlaceholder("hltEnd"))
HLT1ElectronRelaxed = cms.Path(singleElectronL1NonIsolated+cms.SequencePlaceholder("hltEnd"))
HLT2Electron = cms.Path(doubleElectronL1Isolated+cms.SequencePlaceholder("hltEnd"))
HLT2ElectronRelaxed = cms.Path(doubleElectronL1NonIsolated+cms.SequencePlaceholder("hltEnd"))
HLT1Photon = cms.Path(singlePhotonL1Isolated+cms.SequencePlaceholder("hltEnd"))
HLT1PhotonRelaxed = cms.Path(singlePhotonL1NonIsolated+cms.SequencePlaceholder("hltEnd"))
HLT2Photon = cms.Path(doublePhotonL1Isolated+cms.SequencePlaceholder("hltEnd"))
HLT2PhotonRelaxed = cms.Path(doublePhotonL1NonIsolated+cms.SequencePlaceholder("hltEnd"))
HLT1EMHighEt = cms.Path(singleEMHighEtL1NonIsolated+cms.SequencePlaceholder("hltEnd"))
HLT1EMVeryHighEt = cms.Path(singleEMVHEL1NonIsolated+cms.SequencePlaceholder("hltEnd"))
HLT2ElectronZCounter = cms.Path(zeeCounter+cms.SequencePlaceholder("hltEnd"))
HLT2ElectronExclusive = cms.Path(doubleExclElectronL1Isolated+cms.SequencePlaceholder("hltEnd"))
HLT2PhotonExclusive = cms.Path(doubleExclPhotonL1Isolated+cms.SequencePlaceholder("hltEnd"))
HLT1PhotonL1Isolated = cms.Path(singlePhotonPrescaledL1Isolated+cms.SequencePlaceholder("hltEnd"))
CandHLT1ElectronStartup = cms.Path(singleElectronL1IsolatedLargeWindow+cms.SequencePlaceholder("hltEnd"))
CandHLT1ElectronRelaxedStartup = cms.Path(singleElectronL1NonIsolatedLargeWindow+cms.SequencePlaceholder("hltEnd"))
CandHLT2ElectronStartup = cms.Path(doubleElectronL1IsolatedLargeWindow+cms.SequencePlaceholder("hltEnd"))
CandHLT2ElectronRelaxedStartup = cms.Path(doubleElectronL1NonIsolatedLargeWindow+cms.SequencePlaceholder("hltEnd"))


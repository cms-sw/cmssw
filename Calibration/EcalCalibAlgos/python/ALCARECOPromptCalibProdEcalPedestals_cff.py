import FWCore.ParameterSet.Config as cms
import copy
from EventFilter.EcalRawToDigi.EcalUnpackerData_cfi import ecalEBunpacker
from Calibration.EcalCalibAlgos.ecalPedestalPCLworker_cfi import ecalpedestalPCL
from HLTrigger.HLTfilters.hltHighLevel_cfi import *

ALCARECOEcalTestPulsesRaw = copy.deepcopy(hltHighLevel)
ALCARECOEcalTestPulsesRaw.HLTPaths = ['pathALCARECOEcalTestPulsesRaw']
# dont throw on unknown path names
ALCARECOEcalTestPulsesRaw.throw = True
ALCARECOEcalTestPulsesRaw.TriggerResultsTag = cms.InputTag("TriggerResults", "", "RECO")

ALCARECOEcalPedestalsDigis = ecalEBunpacker.clone()
ALCARECOEcalPedestalsDigis.InputLabel = cms.InputTag('hltEcalCalibrationRaw')

ALCARECOEcalPedestals = ecalpedestalPCL.clone()
ALCARECOEcalPedestals.BarrelDigis = cms.InputTag('ALCARECOEcalPedestalsDigis', 'ebDigis')
ALCARECOEcalPedestals.EndcapDigis = cms.InputTag('ALCARECOEcalPedestalsDigis', 'eeDigis')
ALCARECOEcalPedestals.bstRecord   = cms.InputTag('ALCALRECOEcalTCDSDigis', 'bstRecord')


MEtoEDMConvertEcalPedestals = cms.EDProducer("MEtoEDMConverter",
                                             Name=cms.untracked.string('MEtoEDMConverter'),
                                             Verbosity=cms.untracked.int32(0),
                                             # 0 provides no output
                                             # 1 provides basic output
                                             # 2 provide more detailed output
                                             Frequency=cms.untracked.int32(50),
                                             MEPathToSave=cms.untracked.string('AlCaReco/EcalPedestalsPCL'),
                                             deleteAfterCopy=cms.untracked.bool(True)
                                             )

ALCALRECOEcalTCDSDigis = cms.EDProducer('TcdsRawToDigi')
ALCALRECOEcalTCDSDigis.InputLabel =  cms.InputTag('hltEcalCalibrationRaw')

# The actual sequence
seqALCARECOPromptCalibProdEcalPedestals = cms.Sequence(ALCALRECOEcalTCDSDigis    *
                                                       ALCARECOEcalTestPulsesRaw *
                                                       ALCARECOEcalPedestalsDigis *
                                                       ALCARECOEcalPedestals *
                                                       MEtoEDMConvertEcalPedestals)

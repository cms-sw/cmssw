import FWCore.ParameterSet.Config as cms

emap_from_ascii = cms.ESSource("HcalTextCalibrations",
    input = cms.VPSet(cms.PSet(
        object = cms.string('ElectronicsMap'),
        file = cms.FileInPath('official_emap_v5_080208.txt.new_trig')
    ))
)
#es_prefer = cms.ESPrefer("HcalTextCalibrations","emap_from_ascii")

#global configuration
from Configuration.StandardSequences.FrontierConditions_GlobalTag_cff import *
es_prefer_GlobalTag = cms.ESPrefer("PoolDBESSource","GlobalTag")
#off-line
GlobalTag.globaltag = 'CRUZET4_V1::All'
GlobalTag.connect = 'frontier://FrontierProd/CMS_COND_21X_GLOBALTAG'
#on-line
#GlobalTag.globaltag = 'CRZT210_V1H::All'
#GlobalTag.connect = 'frontier://(proxyurl=http://localhost:3128)(serverurl=http://frontier1.cms:8000/FrontierOnProd)(serverurl=http://frontier2.cms:8000/FrontierOnProd)(retrieve-ziplevel=0)/CMS_COND_21X_GLOBALTAG'


#unpacking
from Configuration.StandardSequences.RawToDigi_Data_cff import *

#emulator/comparator
from L1Trigger.HardwareValidation.L1HardwareValidation_cff import *
from L1Trigger.Configuration.L1Config_cff import *

#for LUTs
from DQM.L1TMonitor.Rct_LUTconfiguration_cff import *


#dqm
rctEmulDigis = cms.EDProducer("L1RCTProducer",
    hcalDigis = cms.VInputTag(cms.InputTag("hcalTriggerPrimitiveDigis")),
    useDebugTpgScales = cms.bool(True),
    useEcal = cms.bool(True),
    useHcal = cms.bool(True),
    ecalDigis = cms.VInputTag(cms.InputTag("ecalTriggerPrimitiveDigis")),
    BunchCrossings = cms.vint32(0)                      
)

rctEmulDigis.hcalDigis = cms.VInputTag(cms.InputTag("hcalDigis"))
#rctEmulDigis.ecalDigis=cms.VInputTag(cms.InputTag("ecalEBunpacker"))
rctEmulDigis.ecalDigis = cms.VInputTag(cms.InputTag("ecalDigis:EcalTriggerPrimitives"))

l1tderct = cms.EDAnalyzer("L1TdeRCT",
    rctSourceData = cms.InputTag("l1GctHwDigis"),
    HistFolder = cms.untracked.string('L1TEMU/L1TdeRCT'),
    outputFile = cms.untracked.string('./L1TDQM.root'),
    verbose = cms.untracked.bool(False),
    DQMStore = cms.untracked.bool(True),
    singlechannelhistos = cms.untracked.bool(False),
    ecalTPGData = cms.InputTag("",""),
    rctSourceEmul = cms.InputTag("rctDigis"),
    disableROOToutput = cms.untracked.bool(False),
    hcalTPGData = cms.InputTag("")
)

l1tderct.rctSourceData = 'gctDigis'
l1tderct.rctSourceEmul = 'rctEmulDigis'
#l1tderct.ecalTPGData = 'ecalEBunpacker:EcalTriggerPrimitives'
l1tderct.ecalTPGData = 'ecalDigis:EcalTriggerPrimitives'
l1tderct.hcalTPGData = 'hcalDigis'

l1trct = cms.EDAnalyzer("L1TRCT",
    DQMStore = cms.untracked.bool(True),
    disableROOToutput = cms.untracked.bool(False),
    outputFile = cms.untracked.string('./L1TDQM.root'),
    rctSource = cms.InputTag("l1GctHwDigis","","DQM"),
    verbose = cms.untracked.bool(False)
)

l1trct.rctSource = 'gctDigis'

p = cms.Path(
    cms.SequencePlaceholder("RawToDigi")
    *cms.SequencePlaceholder("rctEmulDigis")
    *cms.SequencePlaceholder("l1trct")
    *cms.SequencePlaceholder("l1tderct")
    )





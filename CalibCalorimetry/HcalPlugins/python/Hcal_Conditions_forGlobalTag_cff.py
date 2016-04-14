import FWCore.ParameterSet.Config as cms

hcal_db_producer = cms.ESProducer("HcalDbProducer",
    dump = cms.untracked.vstring(''),
    file = cms.untracked.string('')
)

#------------------------- HARDCODED conditions

es_hardcode = cms.ESSource("HcalHardcodeCalibrations",
toGet = cms.untracked.vstring('GainWidths'),
#--- the following 5 parameters can be omitted in case of regular Geometry 
    iLumi = cms.double(-1.),                 # for Upgrade: fb-1
    HERecalibration = cms.bool(False),       # True for Upgrade   
    HEreCalibCutoff = cms.double(20.),       # if above is True  
    HFRecalibration = cms.bool(False),       # True for Upgrade
    GainWidthsForTrigPrims = cms.bool(False),# True Upgrade   
    testHFQIE10 = cms.bool(False)            # True 2016
)

from Configuration.StandardSequences.Eras import eras
eras.run2_HF_2016.toModify( es_hardcode, testHFQIE10=cms.bool(True) )

es_prefer_hcalHardcode = cms.ESPrefer("HcalHardcodeCalibrations", "es_hardcode")

def _modifyHcal_Conditions_forGlobalTagForPhase2Common( obj ):
    obj.toGet = cms.untracked.vstring(
                'GainWidths',
                'MCParams',
                'RecoParams',
                'RespCorrs',
                'QIEData',
                'QIETypes',
                'Gains',
                'Pedestals',
                'PedestalWidths',
                'ChannelQuality',
                'ZSThresholds',
                'TimeCorrs',
                'LUTCorrs',
                'LutMetadata',
                'L1TriggerObjects',
                'PFCorrs',
                'ElectronicsMap',
                'CholeskyMatrices',
                'CovarianceMatrices',
                'FlagHFDigiTimeParams'
                )    
    # Special Upgrade trick (if absent - regular case assumed)
    obj.GainWidthsForTrigPrims = cms.bool(True)
    obj.HEreCalibCutoff = cms.double(100.)
    obj.useHBUpgrade = cms.bool(True)
    obj.useHEUpgrade = cms.bool(True)
    obj.useHFUpgrade = cms.bool(True)

from Configuration.StandardSequences.Eras import eras
eras.phase2_common.toModify( es_hardcode,  func=_modifyHcal_Conditions_forGlobalTagForPhase2Common )

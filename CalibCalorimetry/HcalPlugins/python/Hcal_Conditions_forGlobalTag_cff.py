import FWCore.ParameterSet.Config as cms

hcal_db_producer = cms.ESProducer("HcalDbProducer",
    dump = cms.untracked.vstring(''),
    file = cms.untracked.string('')
)

#------------------------- HARDCODED conditions

es_hardcode = cms.ESSource("HcalHardcodeCalibrations",
    toGet = cms.untracked.vstring('GainWidths'),
    iLumi = cms.double(-1.),                 # for Upgrade: fb-1
    iLumiOffset = cms.double(-1.),           # for Upgrade: fb-1
    HBSiPMLumiDep = cms.double(1.7),         # for Upgrade
    HESiPMLumiDep = cms.double(0.7),         # for Upgrade
    HERecalibration = cms.bool(False),       # True for Upgrade   
    HEreCalibCutoff = cms.double(20.),       # if above is True  
    HFRecalibration = cms.bool(False),       # True for Upgrade
    GainWidthsForTrigPrims = cms.bool(False),# True Upgrade   
    useHBUpgrade = cms.bool(False),
    useHEUpgrade = cms.bool(False),
    useHFUpgrade = cms.bool(False),
    testHFQIE10  = cms.bool(False),
    hb = cms.PSet(
        pedestal      = cms.double(3.0),
        pedestalWidth = cms.double(0.55),
        gain          = cms.vdouble(0.19),
        gainWidth     = cms.vdouble(0.0),
        qieType       = cms.int32(0),
        qieOffset     = cms.vdouble(-0.49,1.8,7.2,37.9),
        qieSlope      = cms.vdouble(0.912,0.917,0.922,0.923)
    ),
    he = cms.PSet(
        pedestal      = cms.double(3.0),
        pedestalWidth = cms.double(0.79),
        gain          = cms.vdouble(0.23),
        gainWidth     = cms.vdouble(0),
        qieType       = cms.int32(0),
        qieOffset     = cms.vdouble(-0.38,2.0,7.6,39.6),
        qieSlope      = cms.vdouble(0.912,0.916,0.920,0.922)
    ),
    hf = cms.PSet(
        pedestal      = cms.double(3.0),
        pedestalWidth = cms.double(0.84),
        gain          = cms.vdouble(0.14,0.135),
        gainWidth     = cms.vdouble(0.0,0.0),
        qieType       = cms.int32(0),
        qieOffset     = cms.vdouble(-0.87,1.4,7.8,-29.6),
        qieSlope      = cms.vdouble(0.359,0.358,0.360,0.367)
    ),
    ho = cms.PSet(
        pedestal      = cms.double(11.0),
        pedestalWidth = cms.double(0.57),
        gain          = cms.vdouble(0.0060,0.0087),
        gainWidth     = cms.vdouble(0.0,0.0),
        qieType       = cms.int32(0),
        qieOffset     = cms.vdouble(-0.44,1.4,7.1,38.5),
        qieSlope      = cms.vdouble(0.907,0.915,0.920,0.921)
    ),
    hbUpgrade = cms.PSet(
        pedestal      = cms.double(18.),
        pedestalWidth = cms.double(5.),
        gain          = cms.vdouble(1/900.), #90 is pe/GeV 10 is fC/pe.
        gainWidth     = cms.vdouble(0),
        qieType       = cms.int32(2),
        qieOffset     = cms.vdouble(0.,0.,0.,0.),
        qieSlope      = cms.vdouble(0.333,0.333,0.333,0.333)
    ),
    heUpgrade = cms.PSet(
        pedestal      = cms.double(18.),
        pedestalWidth = cms.double(5.),
        gain          = cms.vdouble(1/900.), #90 is pe/GeV 10 is fC/pe.
        gainWidth     = cms.vdouble(0),
        qieType       = cms.int32(2),
        qieOffset     = cms.vdouble(0.,0.,0.,0.),
        qieSlope      = cms.vdouble(0.333,0.333,0.333,0.333)
    ),
    hfUpgrade = cms.PSet(
        pedestal      = cms.double(13.33),
        pedestalWidth = cms.double(3.33),
        gain          = cms.vdouble(0.14,0.135),
        gainWidth     = cms.vdouble(0.0,0.0),
        qieType       = cms.int32(1),
        qieOffset     = cms.vdouble(0.0697,-0.7405,12.38,-671.9),
        qieSlope      = cms.vdouble(0.297,0.298,0.298,0.313)
    ),
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

#
import FWCore.ParameterSet.Config as cms

process = cms.Process("d")

import HLTrigger.HLTfilters.hltHighLevel_cfi as hlt
# accept if 'path_1' succeeds
process.hltfilter = hlt.hltHighLevel.clone(
# Min-Bias
#    HLTPaths = ['HLT_Physics_v*'],
#    HLTPaths = ['HLT_L1Tech_BSC_minBias_threshold1_v*'],
    HLTPaths = ['HLT_Random_v*'],
#    HLTPaths = ['HLT_ZeroBias_v*'],
# Commissioning:
#    HLTPaths = ['HLT_L1_Interbunch_BSC_v*'],
#    HLTPaths = ['HLT_L1_PreCollisions_v*'],
#    HLTPaths = ['HLT_BeamGas_BSC_v*'],
#    HLTPaths = ['HLT_BeamGas_HF_v*'],
# examples
#    HLTPaths = ['p*'],
#    HLTPaths = ['path_?'],
    andOr = True,  # False = and, True=or
    throw = False
    )


# process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('dumper'),
    destinations = cms.untracked.vstring('cout'),
#    destinations = cms.untracked.vstring("log","cout"),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING')
    )
#    log = cms.untracked.PSet(
#        threshold = cms.untracked.string('DEBUG')
#    )
)

#process.MessageLogger.cerr.FwkReport.reportEvery = 1
#process.MessageLogger.cerr.threshold = 'Debug'

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(                          
# "/store/data/Commissioning11/Cosmics/RAW/v1/000/157/884/FE867543-D938-E011-B9C4-001D09F2983F.root",
# 186160
# "rfio:/castor/cern.ch/cms/store/data/Commissioning12/Cosmics/RAW/v1/000/186/160/0224F8A5-6A61-E111-BCC5-BCAEC5329708.root"
# "rfio:/castor/cern.ch/cms/store/data/Commissioning12/Commissioning/RAW/v1/000/186/160/04D2072A-8161-E111-9058-BCAEC518FF5A.root"
# "rfio:/castor/cern.ch/cms/store/data/Commissioning12/MinimumBias/RAW/v1/000/186/160/60406E4F-8A61-E111-9CF3-BCAEC53296F8.root",
# "rfio:/castor/cern.ch/cms/store/data/Commissioning12/MinimumBias/RAW/v1/000/186/160/62193B74-6D61-E111-8044-BCAEC518FF7A.root",
# "rfio:/castor/cern.ch/cms/store/data/Commissioning12/MinimumBias/RAW/v1/000/186/160/847DCF6E-8061-E111-9638-BCAEC5329708.root",
# "rfio:/castor/cern.ch/cms/store/data/Commissioning12/MinimumBias/RAW/v1/000/186/160/9C639109-9E61-E111-BC15-E0CB4E55365C.root",
# "rfio:/castor/cern.ch/cms/store/data/Commissioning12/MinimumBias/RAW/v1/000/186/160/A4420D9C-7661-E111-8763-5404A63886BD.root",
# "rfio:/castor/cern.ch/cms/store/data/Commissioning12/MinimumBias/RAW/v1/000/186/160/EC5A04E5-9461-E111-8600-BCAEC53296F5.root",
# "rfio:/castor/cern.ch/cms/store/data/Commissioning12/MinimumBias/RAW/v1/000/186/160/F2915E5C-A461-E111-8FBF-BCAEC532970D.root"
# 185984
# "rfio:/castor/cern.ch/cms/store/data/Commissioning12/Commissioning/RAW/v1/000/185/984/00DB308B-775E-E111-94B1-BCAEC532972E.root"
# 185250
# "rfio:/castor/cern.ch/cms/store/data/Commissioning12/Commissioning/RAW/v1/000/185/250/0286CCB5-7858-E111-B19E-BCAEC5329708.root"

# 186822
# "rfio:/castor/cern.ch/cms/store/data/Commissioning12/MinimumBias/RAW/v1/000/186/822/06D621F9-1C68-E111-817A-001D09F2906A.root",
    
# 186996
# "rfio:/castor/cern.ch/cms/store/data/Commissioning12/MinimumBias/RAW/v1/000/186/996/1A8BFCC6-BE68-E111-B0F5-0019B9F72F97.root",
# "rfio:/castor/cern.ch/cms/store/data/Commissioning12/MinimumBias/RAW/v1/000/186/996/1E2C2DC4-B468-E111-B1C6-001D09F25267.root",
# "rfio:/castor/cern.ch/cms/store/data/Commissioning12/MinimumBias/RAW/v1/000/186/996/1ECD78E8-D068-E111-9B5F-001D09F29321.root",
# "rfio:/castor/cern.ch/cms/store/data/Commissioning12/MinimumBias/RAW/v1/000/186/996/2A2E8E83-A968-E111-97A0-001D09F24399.root",
# "rfio:/castor/cern.ch/cms/store/data/Commissioning12/MinimumBias/RAW/v1/000/186/996/523DC559-EC68-E111-9534-0019B9F4A1D7.root",
# "rfio:/castor/cern.ch/cms/store/data/Commissioning12/MinimumBias/RAW/v1/000/186/996/5E323AB2-E668-E111-B4FA-001D09F295FB.root",
# "rfio:/castor/cern.ch/cms/store/data/Commissioning12/MinimumBias/RAW/v1/000/186/996/62F17EC4-C768-E111-A53F-001D09F23D1D.root",
# "rfio:/castor/cern.ch/cms/store/data/Commissioning12/MinimumBias/RAW/v1/000/186/996/7A1B5B90-DD68-E111-AB69-001D09F2983F.root",
# "rfio:/castor/cern.ch/cms/store/data/Commissioning12/MinimumBias/RAW/v1/000/186/996/A045EE58-A068-E111-8DCF-001D09F27003.root",

# "rfio:/castor/cern.ch/cms/store/data/Commissioning12/Cosmics/RAW/v1/000/186/996/007B8341-C468-E111-9909-001D09F2906A.root",

# "rfio:/castor/cern.ch/cms/store/data/Commissioning12/Commissioning/RAW/v1/000/186/996/006860ED-D068-E111-8327-001D09F24303.root",

# 187361
 "rfio:/castor/cern.ch/cms/store/data/Commissioning12/MinimumBias/RAW/v1/000/187/361/483A4D50-566A-E111-AE98-001D09F2AF96.root",
 "rfio:/castor/cern.ch/cms/store/data/Commissioning12/MinimumBias/RAW/v1/000/187/361/5630B3CF-6C6A-E111-BA18-001D09F28F25.root",
 "rfio:/castor/cern.ch/cms/store/data/Commissioning12/MinimumBias/RAW/v1/000/187/361/6CD5D083-806A-E111-BEDD-001D09F2906A.root",
 "rfio:/castor/cern.ch/cms/store/data/Commissioning12/MinimumBias/RAW/v1/000/187/361/70C71F71-856A-E111-839E-001D09F2983F.root",
 "rfio:/castor/cern.ch/cms/store/data/Commissioning12/MinimumBias/RAW/v1/000/187/361/A6868F2F-676A-E111-AD64-001D09F2305C.root",
 "rfio:/castor/cern.ch/cms/store/data/Commissioning12/MinimumBias/RAW/v1/000/187/361/B465D505-5E6A-E111-952C-001D09F29114.root",
 "rfio:/castor/cern.ch/cms/store/data/Commissioning12/MinimumBias/RAW/v1/000/187/361/F6994FC3-786A-E111-99FA-001D09F25267.root",
 "rfio:/castor/cern.ch/cms/store/data/Commissioning12/MinimumBias/RAW/v1/000/187/361/F6BA5695-746A-E111-B8EF-001D09F2910A.root",

# 187446
# "rfio:/castor/cern.ch/cms/store/data/Commissioning12/MinimumBias/RAW/v1/000/187/446/02CC6857-376B-E111-ADC2-001D09F28E80.root",
# "rfio:/castor/cern.ch/cms/store/data/Commissioning12/MinimumBias/RAW/v1/000/187/446/0438B1B9-826B-E111-919B-001D09F24D8A.root",
# "rfio:/castor/cern.ch/cms/store/data/Commissioning12/MinimumBias/RAW/v1/000/187/446/065A1CA7-3D6B-E111-81A4-001D09F24FEC.root",
# "rfio:/castor/cern.ch/cms/store/data/Commissioning12/MinimumBias/RAW/v1/000/187/446/088CEEC5-2C6B-E111-9C1D-001D09F24D67.root",
# "rfio:/castor/cern.ch/cms/store/data/Commissioning12/MinimumBias/RAW/v1/000/187/446/12549E7F-916B-E111-8972-001D09F2B30B.root",
# "rfio:/castor/cern.ch/cms/store/data/Commissioning12/MinimumBias/RAW/v1/000/187/446/146B7135-546B-E111-8E3E-0019B9F72F97.root",
# "rfio:/castor/cern.ch/cms/store/data/Commissioning12/MinimumBias/RAW/v1/000/187/446/1630241C-A36B-E111-ADC0-003048D374F2.root",
# "rfio:/castor/cern.ch/cms/store/data/Commissioning12/MinimumBias/RAW/v1/000/187/446/16B7554B-436B-E111-BAEC-001D09F251FE.root",
# "rfio:/castor/cern.ch/cms/store/data/Commissioning12/MinimumBias/RAW/v1/000/187/446/1883852F-4D6B-E111-ACF5-001D09F241F0.root",
# "rfio:/castor/cern.ch/cms/store/data/Commissioning12/MinimumBias/RAW/v1/000/187/446/24FCE29B-1C6B-E111-928D-001D09F2910A.root",
 

    )

)

#process.out = cms.OutputModule("PoolOutputModule",
#    fileName =  cms.untracked.string('file:histos.root')
#)

process.dumper = cms.EDAnalyzer("findHotPixels", 
    Timing = cms.untracked.bool(False),
    IncludeErrors = cms.untracked.bool(True),
#    InputLabel = cms.untracked.string('rawDataCollector'),
#   In 2012, extension = _LHC                                
    InputLabel = cms.untracked.string('rawDataCollector'),
#    InputLabel = cms.untracked.string('siPixelRawData'),
    CheckPixelOrder = cms.untracked.bool(False)
)

process.p = cms.Path(process.hltfilter*process.dumper)
# process.p = cms.Path(process.dumper)

# process.ep = cms.EndPath(process.out)



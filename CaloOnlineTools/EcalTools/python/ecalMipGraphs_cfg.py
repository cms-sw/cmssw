import FWCore.ParameterSet.Config as cms

process = cms.Process("ECALMIPGRAPHS")
process.load("EventFilter.EcalRawToDigi.EcalUnpackerMapping_cfi")

process.load("EventFilter.EcalRawToDigi.EcalUnpackerData_cfi")

process.load("Geometry.CaloEventSetup.CaloTopology_cfi")

process.load("Geometry.EcalCommonData.EcalOnly_cfi")

process.load("Geometry.CaloEventSetup.CaloGeometry_cff")
process.load("Geometry.EcalMapping.EcalMapping_cfi")
process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")

process.load("CalibCalorimetry.EcalLaserCorrection.ecalLaserCorrectionService_cfi")

import RecoLocalCalo.EcalRecProducers.ecalMaxSampleUncalibRecHit_cfi
process.ecalUncalibHit = RecoLocalCalo.EcalRecProducers.ecalMaxSampleUncalibRecHit_cfi.ecalMaxSampleUncalibRecHit.clone()
process.load("RecoLocalCalo.EcalRecProducers.ecalRecHit_cfi")

process.load("CaloOnlineTools.EcalTools.ecalMipGraphs_cfi")
process.load("HLTrigger.special.TriggerTypeFilter_cfi")

process.source = cms.Source("PoolSource",
    skipEvents = cms.untracked.uint32(0),
    fileNames = cms.untracked.vstring(#'/store/data/Commissioning08/Cosmics/RAW/CRUZET4_v1/000/057/289/1E1407F1-106D-DD11-97A7-000423D985E4.root'
#'/store/data/Commissioning08/Cosmics/RAW/CRUZET4_v1/000/058/359/005A40D9-1470-DD11-A2B6-001617C3B6DE.root')
'/store/data/Commissioning08/Cosmics/RAW/CRUZET4_v1/000/057/771/00D18762-386E-DD11-A081-0016177CA7A0.root')
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.load("CalibCalorimetry.EcalTrivialCondModules.EcalTrivialCondRetriever_cfi")

process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.GlobalRuns.ForceZeroTeslaField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.dumpEv = cms.EDAnalyzer("EventContentAnalyzer")

process.MessageLogger = cms.Service("MessageLogger",
    #suppressInfo = cms.untracked.vstring('ecalEBunpacker'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG')
    ),
    categories = cms.untracked.vstring('EcalMipGraphs'),
    destinations = cms.untracked.vstring('cout')
)

process.TFileService = cms.Service("TFileService",
  fileName = cms.string('ecalMipGraphs-57771.graph.root'),
  closeFileFast = cms.untracked.bool(True)
)

process.p = cms.Path(process.triggerTypeFilter*process.ecalEBunpacker*process.ecalUncalibHit*process.ecalRecHit*process.ecalMipGraphs)

process.GlobalTag.globaltag = 'CRUZET4_V1P::All'

process.ecalUncalibHit.EBdigiCollection = 'ecalEBunpacker:ebDigis'
process.ecalUncalibHit.EEdigiCollection = 'ecalEBunpacker:eeDigis'
process.ecalRecHit.ChannelStatusToBeExcluded = [1]
process.ecalRecHit.EBuncalibRecHitCollection = 'ecalUncalibHit:EcalUncalibRecHitsEB'
process.ecalRecHit.EEuncalibRecHitCollection = 'ecalUncalibHit:EcalUncalibRecHitsEE'
process.EcalTrivialConditionRetriever.producedEcalWeights = False
process.EcalTrivialConditionRetriever.producedEcalPedestals = False
process.EcalTrivialConditionRetriever.producedEcalIntercalibConstants = False
process.EcalTrivialConditionRetriever.producedEcalIntercalibErrors = False
process.EcalTrivialConditionRetriever.producedEcalGainRatios = False
process.EcalTrivialConditionRetriever.producedEcalADCToGeVConstant = False
process.EcalTrivialConditionRetriever.producedEcalLaserCorrection = False
process.EcalTrivialConditionRetriever.producedChannelStatus = cms.untracked.bool(False)
#process.EcalTrivialConditionRetriever.channelStatusFile = 'CaloOnlineTools/EcalTools/data/listCRUZET4.v2.hashed.txt'
#es_prefer_EcalChannelStatus = cms.ESPrefer("EcalTrivialConditionRetriever","EcalChannelStatus")
#process.ecalUncalibHit.EBdigiCollection = 'ecalEBunpacker:ebDigis'
#process.ecalUncalibHit.EEdigiCollection = 'ecalEBunpacker:eeDigis'
#process.ecalRecHit.ChannelStatusToBeExcluded = [1]
#process.ecalRecHit.EBuncalibRecHitCollection = 'ecalUncalibHit:EcalUncalibRecHitsEB'
#process.ecalRecHit.EEuncalibRecHitCollection = 'ecalUncalibHit:EcalUncalibRecHitsEE'
process.ecalMipGraphs.amplitudeThreshold = 0.5
process.triggerTypeFilter.SelectedTriggerType = 1

#list of hot channels
#process.ecalMipGraphs.seedCrys = (27034,27033,27032,27031,27030,27390,27391,27392,27393,27394,27754,27753,27752,27751,27750,28110,28111,28112,28113,28114,28474,28473,28471,28470,27039,27038,27037,27036,27035,27395,27396,27397,27398,27399,27759,27758,27757,27756,27755,28115,28116,28117,28118,28119,28479,28478,28477,28476,28475,19884,19883,19882,19881,19880,20240,20241,20242,20243,20244,20604,20603,20602,20601,20600,20960,20961,20962,20963,20964,21324,21323,21322,21321,21320,19889,19888,19887,19886,19885,20245,20246,20247,20248,20249,20609,20608,20607,20606,20605,20965,20966,20967,20968,20969,21329,21328,21327,21326,21325,3330,3331,3332,3333,3334,2974,2973,2972,2971,2970,2610,2611,2612,2613,2614,2254,2253,2252,2251,2250,1890,1891,1892,1893,1894,3335,3336,3337,3338,3339,2979,2978,2977,2976,2975,2615,2616,2617,2618,2619,2259,2258,2257,2256,2255,1895,1897,1898,1899,10580,10581,10582,10583,10584,10224,10223,10222,10220,9860,9861,9862,9863,9864,9504,9503,9502,9501,9500,9140,9141,9142,9143,9144,10585,10586,10587,10588,10589,10229,10228,10227,10226,10225,9865,9866,9868,9869,9509,9508,9507,9506,9505,9145,9146,9147,9148,9149,41155,41156,41157,41158,41159,40799,40798,40797,40796,40795,40435,40436,40437,40438,40439,40079,40078,40077,40076,40075,39715,39716,39718,39719,41150,41151,41152,41153,41154,40794,40793,40792,40791,40790,40430,40431,40432,40433,40434,40074,40073,40072,40071,40070,39710,39711,39712,39713,39714,57555,57556,57557,57558,57559,57199,57198,57197,57196,57195,56835,56836,56837,56838,56839,56479,56478,56477,56476,56475,56115,56116,56117,56118,56119,57550,57551,57552,57553,57554,57194,57193,57192,57191,57190,56830,56831,56832,56833,56834,56474,56472,56471,56470,56110,56111,56112,56113,56114,36329,36328,36327,36326,36325,36685,36686,36687,36688,36689,37049,37048,37047,37046,37045,37405,37406,37407,37408,37409,37769,37768,37767,37766,37765,36324,36323,36322,36321,36320,36680,36681,36682,36683,36684,37044,37043,37042,37041,37040,37400,37401,37402,37403,37404,37764,37763,37762,37761,37760,141,199,276,277,297,414,559,999,1358,1896,2847,3049,3241,3242,3243,3713,3817,4176,4177,4263,5532,6932,6972,6978,7151,7152,7169,7507,8108,8583,9867,10070,10221,10902,12256,13024,13485,14314,14502,14528,14636,14639,14663,14673,14675,15193,15323,15808,16002,16210,16675,17107,17255,20005,20040,21013,21300,21446,21740,21803,22086,22437,23286,24060,24317,24491,26832,27877,28410,28472,30079,30425,30501,32006,32564,32680,32681,32682,32683,32684,32705,33040,33041,33042,33043,33044,33101,33280,33400,33401,33402,33403,33404,33760,33761,33762,33763,33764,33964,34120,34121,34122,34123,34124,34267,34853,36669,39296,39392,39665,39717,39798,39799,39947,40247,40306,42125,42407,42769,42803,42805,43127,43980,45736,45775,45784,46144,46456,46478,46504,46506,46555,46655,46712,46845,46865,46875,46889,47137,49189,49217,49392,49867,50466,51628,52052,52058,53802,53825,53845,54073,54145,54659,54863,55038,55152,56264,56473,56818,57958,58746,60319,60679,60884,61058,61195,1,674,1413,1496,1497,2910,2911,2912,2927,2928,2929,3010,3011,3028,3029,3110,3129,4610,4629,4710,4711,4728,4729,4810,4811,4812,4827,4828,4829,6242,6243,6252,6321,6326,6327,7767,7812,7842,8063,8120,8121,8122,8123,8487,9072,9074,9075,9146,9158,9159,9230,9320,10650,10651,10652,10667,10668,10669,10750,10751,10768,10769,10850,10869,12350,12369,12450,12451,12468,12469,12550,12551,12552,12567,12568,12569,12586,12587,14061) # if you want to fix on one xtal (hashed Index)

#process.ecalMipGraphs.maskedFEDs = (601, 602, 603, 604, 605, 606, 607, 608, 609,646,647,648,649,650,651,652,653,654)

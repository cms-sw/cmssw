import FWCore.ParameterSet.Config as cms

process = cms.Process("CREATESQLITE")
process.load("CondCore.CondDB.CondDB_cfi")
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['run2_design']

process.CondDB.connect = 'sqlite_file:MuonSystemAging.db'

process.source = cms.Source("EmptyIOVSource",
    lastValue = cms.uint64(1),
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    interval = cms.uint64(1)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDB,
    timetype = cms.untracked.string('runnumber'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('MuonSystemAgingRcd'),
        tag = cms.string('MuonSystemAging_test')
    ))
)

process.produceAgingObject = cms.EDAnalyzer("ProduceAgingObject",

            maskedGEMIDs = cms.vint32([
            671105280,671121664,671105792,671122176,671106304,671122688,671106816,
            671123200,671107328,671123712,671107840,671124224,671108352,671124736,
            671105794,671122178,671105282,671121666,671106306,671122690,671106818,
            671123202,671107330,671123714,671107842,671124226,671108354,671124738,
            671105056, 671121440, 671105568, 671121952, 671106080, 671122464, 
            671106592, 671122976, 671107104, 671123488, 671107616, 671124000, 
            671105058, 671121442, 671105570, 671121954, 671106082, 671122466, 
            671106594, 671122978, 671107106, 671123490, 671107618, 671124002, 
            ]),

            # GE11MinusIDs
            # cms.vint32([
            # 671105280,671121664,671105792,671122176,671106304,671122688,671106816,
            # 671123200,671107328,671123712,671107840,671124224,671108352,671124736,
            # 671108864,671125248,671109376,671125760,671109888,671126272,671110400,
            # 671126784,671110912,671127296,671111424,671127808,671111936,671128320,
            # 671112448,671128832,671112960,671129344,671113472,671129856,671113984,
            # 671130368,671105024,671121408,671105536,671121920,671106048,671122432,
            # 671106560,671122944,671107072,671123456,671107584,671123968,671108096,
            # 671124480,671108608,671124992,671109120,671125504,671109632,671126016,
            # 671110144,671126528,671110656,671127040,671111168,671127552,671111680,
            # 671128064,671112192,671128576,671112704,671129088,671113216,671129600,
            # 671113728,671130112
            # ]),

            # GE11PlusIDs 
            # cms.vint32([
            # 671105794,671122178,671105282,671121666,671106306,671122690,671106818,
            # 671123202,671107330,671123714,671107842,671124226,671108354,671124738,
            # 671108866,671125250,671109378,671125762,671109890,671126274,671110402,
            # 671126786,671110914,671127298,671111426,671127810,671111938,671128322,
            # 671112450,671128834,671112962,671129346,671113474,671129858,671113986,
            # 671130370,671105026,671121410,671105538,671121922,671106050,671122434,
            # 671106562,671122946,671107074,671123458,671107586,671123970,671108098,
            # 671124482,671108610,671124994,671109122,671125506,671109634,671126018,
            # 671110146,671126530,671110658,671127042,671111170,671127554,671111682,
            # 671128066,671112194,671128578,671112706,671129090,671113218,671129602,
            # 671113730,671130114
            # ]),

            # GE21MinusIDs 
            # cms.vint32([
            # 671105056, 671121440, 671105568, 671121952, 671106080, 671122464, 
            # 671106592, 671122976, 671107104, 671123488, 671107616, 671124000, 
            # 671108128, 671124512, 671108640, 671125024, 671109152, 671125536, 
            # 671105312, 671121696, 671105824, 671122208, 671106336, 671122720, 
            # 671106848, 671123232, 671107360, 671123744, 671107872, 671124256, 
            # 671108384, 671124768, 671108896, 671125280, 671109408, 671125792
            # ]),

            # GE21PlusIDs 
            # cms.vint32([
            # 671105058, 671121442, 671105570, 671121954, 671106082, 671122466, 
            # 671106594, 671122978, 671107106, 671123490, 671107618, 671124002, 
            # 671108130, 671124514, 671108642, 671125026, 671109154, 671125538, 
            # 671105314, 671121698, 671105826, 671122210, 671106338, 671122722, 
            # 671106850, 671123234, 671107362, 671123746, 671107874, 671124258, 
            # 671108386, 671124770, 671108898, 671125282, 671109410, 671125794
            # ]),

            maskedME0IDs = cms.vint32([]),
            
            # Accept lists or regular expression as from:
            # http://www.cplusplus.com/reference/regex/ECMAScript/
            dtRegEx = cms.vstring([
            # A chamber by chamber list in format CHAMBER:EFF

            # MB4 of top sectors with EFF = 0
            "WH-2_ST4_SEC2$:0.","WH-2_ST4_SEC3$:0.","WH-2_ST4_SEC4$:0.",
            "WH-2_ST4_SEC5$:0.","WH-2_ST4_SEC6$:0.","WH-1_ST4_SEC2$:0.",
            "WH-1_ST4_SEC3$:0.","WH-1_ST4_SEC4$:0.","WH-1_ST4_SEC5$:0.",
            "WH-1_ST4_SEC6$:0.","WH0_ST4_SEC2$:0.","WH0_ST4_SEC3$:0.",
            "WH0_ST4_SEC4$:0.","WH0_ST4_SEC5$:0.","WH0_ST4_SEC6$:0.",
            "WH1_ST4_SEC2$:0.","WH1_ST4_SEC3$:0.","WH1_ST4_SEC4$:0.",
            "WH1_ST4_SEC5$:0.","WH1_ST4_SEC6$:0.","WH2_ST4_SEC2$:0.",
            "WH2_ST4_SEC3$:0.","WH2_ST4_SEC4$:0.","WH2_ST4_SEC5$:0.",
            "WH2_ST4_SEC6$:0.",
            # MB1 of external wheels with EFF = 0
            "WH-2_ST1_SEC1$:0.","WH-2_ST1_SEC2$:0.","WH-2_ST1_SEC3$:0.",
            "WH-2_ST1_SEC4$:0.","WH-2_ST1_SEC5$:0.","WH-2_ST1_SEC6$:0.",
            "WH-2_ST1_SEC7$:0.","WH-2_ST1_SEC8$:0.","WH-2_ST1_SEC9$:0.",
            "WH-2_ST1_SEC10$:0.","WH-2_ST1_SEC11$:0.","WH-2_ST1_SEC12$:0.",
            "WH2_ST1_SEC1$:0.","WH2_ST1_SEC2$:0.","WH2_ST1_SEC3$:0.",
            "WH2_ST1_SEC4$:0.","WH2_ST1_SEC5$:0.","WH2_ST1_SEC6$:0.",
            "WH2_ST1_SEC7$:0.","WH2_ST1_SEC8$:0.","WH2_ST1_SEC9$:0.",
            "WH2_ST1_SEC10$:0.","WH2_ST1_SEC11$:0.","WH2_ST1_SEC12$:0.",
            # 5 MB2s of external wheels with EFF = 0
            "WH2_ST2_SEC3$:0.","WH2_ST2_SEC6$:0.","WH2_ST2_SEC9$:0.",
            "WH-2_ST2_SEC2$:0.","WH-2_ST2_SEC4$:0.",
            # more sparse failures with EFF = 0
            "WH-2_ST2_SEC8$:0.","WH-1_ST1_SEC1$:0.","WH-1_ST2_SEC1$:0.",
            "WH-1_ST1_SEC4$:0.","WH-1_ST3_SEC7$:0.","WH0_ST2_SEC2$:0.",
            "WH0_ST3_SEC5$:0.","WH0_ST4_SEC12$:0.","WH1_ST1_SEC6$:0.",
            "WH1_ST1_SEC10$:0.","WH1_ST3_SEC3$:0."

            # Or a RegEx setting efficiency  for all chamber to 10%
            #"(WH-?\\\d_ST\\\d_SEC\\\\d+):0.1"
            ]),
            
            cscRegEx = cms.vstring([
            # # Set 70% type-2 efficiency on ME-1
            # "(ME[-]1/\\\d/\\\\d+):2,0.7",

            # # Set 30% type-1 efficiency on ME- endcap
            # "(-\\\d/\\\d/\\\\d+):1,0.3",

            # type-xy efficiency: x is layer (0 for chamber), 
            # y = 0,1,2 for all digis, strip digis, wire digis
            # Set 0% type-1 efficiency on ME+1/1/10A --> No strip digis
            # "ME\\\+1/4/10:1,0.0",
            # Set 0% type-1 efficiency on ME+1/1/10B --> No strip digis
            "ME\\\+1/1/10:1,0.0",
            # Set 0% type-0 efficiency on ME+1/2/4 --> No digis
            "ME\\\+1/2/4:0,0.0",
            # Set 0% type-2 efficiency on ME+1/2/15 --> No wire digis
            "ME\\\+1/2/15:2,0.0",
            # Set 0% type-31 efficiency on ME+1/2/26 --> No digis on layer 3
            "ME\\\+1/2/26:30,0.0", 
            # Set 0% type-31 efficiency on ME-1/2/7 --> No strip digis on layer 3
            "ME\\\-1/2/7:31,0.0",
            # Set 50% type-32 efficiency on ME-1/2/18 --> 50% wire digis on layer 3
            "ME\\\-1/2/18:32,0.5",
            ]),

           rpcRegEx = cms.vstring(["637570221:0.0","637602989:0.5","637569561:0.7"])


)

process.p = cms.Path(process.produceAgingObject)


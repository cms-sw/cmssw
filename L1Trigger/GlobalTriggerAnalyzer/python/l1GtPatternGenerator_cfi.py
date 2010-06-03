import FWCore.ParameterSet.Config as cms


l1GtPatternGenerator = cms.EDAnalyzer("L1GtPatternGenerator",
    # input tags for various records
    GtInputTag = cms.InputTag("gtDigis"),
    GmtInputTag = cms.InputTag("gmtDigis"),
    GctInputTag = cms.InputTag("gctDigis"),
    CscInputTag = cms.InputTag("gtDigis", "CSC"),
    DtInputTag  = cms.InputTag("gtDigis", "DT"),
    RpcbInputTag  = cms.InputTag("gtDigis", "RPCb"),
    RpcfInputTag  = cms.InputTag("gtDigis", "RPCf"),   

    # file name
    PatternFileName = cms.string("GT_GMT_patterns.txt"),

    # bunch crossing numbers to write
    bx = cms.vint32(0),

    # header
    PatternFileHeader = cms.string(
"""#GT_GMT_patterns_VD
#  
# editors - HB 220606
#  
# remarks:
# values in this template are for version VD (same as VB) for the cond-chips of GTL9U (from IVAN)
#
# syntax:
# character "#" indicates a comment line
# header line 1 => hardware of sim- and spy-memories
# header line 2 => hardware location (FPGA-chip) of sim-memories
# header line 3 => channel number of sim-memories (PSB)
# header line 4 => hardware location (FPGA-chip) of spy-memories
# header line 5 => name of patterns
# header line 6 => number of objects (calos, muons) or other declarations
# (header line 7 => only graphics)
# (header line 8 => only text and graphics)
# header line 9 => number of columns, starting with 0
#
# patterns:
# values in column 0 are event numbers (decimal), starting with 0 (synchronisation data)
# patterns for 1024 events (memories of cond-chips on GTL9U can contain only 1024 events) are in this file
# values in columns 1-119 are the hexadecimal patterns, the rightmost digit in a string is LSB
#
# header:
# e |<--------------------------------------------------------------------------PSB/GTL9U(REC)------------------------------------------------------------------------------------------------------------->|<--------------------------------------------------------------------------PSB/GMT(AUF,AUB)--------------------------------------------------------------------------------------------------------------------------------------------------->|<----------------------------------------------------------------GMT REGIONAL MUONs----------------------------------------------------------->|<----GMT(SORT)/GTL9U(REC)----->|<--------------GTL9U(COND)/FDL(ALGO)---------------->|<-----------FDL----------->|
# v |PSB slot13/ch6+7   |PSB slot13/ch4+5   |PSB slot13/ch2+3   |PSB slot13/ch0+1   |PSB slot14/ch6+7   |PSB slot14/ch4+5   |PSB slot14/ch2+3   |PSB slot14/ch0+1   |PSB slot15/ch2+3   |PSB slot15/ch0+1   |PSB slot19/ch6+7   |PSB slot19/ch4+5   |PSB slot19/ch2+3   |PSB slot19/ch0+1   |PSB slot20/ch6+7   |PSB slot20/ch4+5   |PSB slot20/ch2+3   |PSB slot20/ch0+1   |PSB slot21/ch6+7   |PSB slot21/ch4+5   |PSB slot21/ch2+3   |PSB slot21/ch0+1   |GMT INF                            |GMT INC                            |GMT IND                            |GMT INB                            |GMT SORT                       |COND1                     |COND2                     |PSB slot9/ch0+1    |FINOR  |
# e |ch6  ch7  ch6  ch7 |ch4  ch5  ch4  ch5 |ch2  ch3  ch2  ch3 |ch0  ch1  ch0  ch1 |ch6  ch7  ch6  ch7 |ch4  ch5  ch4  ch5 |ch2  ch3  ch2  ch3 |ch0  ch1  ch0  ch1 |ch2  ch3  ch2  ch3 |ch0  ch1  ch0  ch1 |ch6  ch7  ch6  ch7 |ch4  ch5  ch4  ch5 |ch2  ch3  ch2  ch3 |ch0  ch1  ch0  ch1 |ch6  ch7  ch6  ch7 |ch4  ch5  ch4  ch5 |ch2  ch3  ch2  ch3 |ch0  ch1  ch0  ch1 |ch6  ch7  ch6  ch7 |ch4  ch5  ch4  ch5 |ch2  ch3  ch2  ch3 |ch0  ch1  ch0  ch1 |                                   |                                   |                                   |                                   |                               |                          |                          |ch0  ch1  ch0  ch1 |       |
# n |GTL9U REC1         |GTL9U REC1         |GTL9U REC2         |GTL9U REC2         |GTL9U REC2         |GTL9U REC2         |GTL9U REC3         |GTL9U REC3         |GTL9U REC3         |GTL9U REC3         |GMT AUF            |GMT AUF            |GMT AUB            |GMT AUB            |GMT AUF            |GMT AUF            |GMT AUB            |GMT AUB            |GMT AUF            |GMT AUF            |GMT AUB            |GMT AUB            |                                   |                                   |                                   |                                   |GTL9U REC1                     |FDL ALGO                  |FDL ALGO                  |FDL ALGO           |       |
# t |calo1 (ieg)        |calo2 (eg)         |calo3 (jet)        |calo4 (fwdjet)     |calo5 (tau)        |calo6 (esums)      |calo7 (hfbc/etsums)|calo8 (free)       |calo9 (totem)      |calo10 (free)      |MQF4               |MQF3               |MQB2               |MQB1               |MQF8               |MQF7               |MQB6               |MQB5               |MQF12              |MQF11              |MQB10              |MQB9               |RPC forward                        |CSC                                |DT                                 |RPC barrel                         |muon (sorted four)             |algo                      |algo                      |techtrigger        |       |
#   | 1    2    3    4  | 1    2    3    4  | 1    2    3    4  | 1    2    3    4  | 1    2    3    4  | 1    2    3    4  | 1    2    3    4  | 1    2    3    4  | 1    2    3    4  | 1    2    3    4  |45M  45Q   6M   6Q |45M  45Q   6M   6Q |01M  01Q  23M  23Q |01M  01Q  23M  23Q |45M  45Q   6M   6Q |45M  45Q   6M   6Q |01M  01Q  23M  23Q |01M  01Q  23M  23Q |45M  45Q   6M   6Q |45M  45Q   6M   6Q |01M  01Q  23M  23Q |01M  01Q  23M  23Q |   1        2        3        4    |   1        2        3        4    |   1        2        3        4    |   1        2        3        4    |   1       2       3       4   |191--160 159--128 127---96|95----64 63----32 31-----0|15-0 47-32 31-16 63-48|    |
#   |                   |                   |                   |                   |                   |                   |                   |                   |                   |                   |                   |                   |                   |                   |                   |                   |                   |                   |                   |                   |                   |                   |                                   |                                   |                                   |                                   |                               |                          |                          |                   |       |
# columns:              |                   |                   |                   |                   |                   |                   |                   |                   |                   |                   |                   |                   |                   |                   |                   |                   |                   |                   |                   |                   |                   |                                   |                                   |                                   |                                   |                               |                          |                          |                   |       |
# 0 | 1    2    3    4  | 5    6    7    8  | 9    10   11   12 | 13   14   15   16 | 17   18   19   20 | 21   22   23   24 | 25   26   27   28 | 29   30   31   32 | 33   34   35   36 | 37   38   39   40 | 41   42   43   44 | 45   46   47   48 | 49   50   51   52 | 53   54   55   56 | 57   58   59   60 | 61   62   63   64 | 65   66   67   68 | 69   70   71   72 | 73   74   75   76 | 77   78   79   80 | 81   82   83   84 | 85   86   87   88 |   89       90       91       92   |   93       94       95       96   |   97       98       99      100   |  101      102      103      104   |  105     106     107     108  |  109      110      111   |   112      113      114  | 115  116  117  118|119    |
"""),

   # footer                                      
   PatternFileFooter = cms.string(""),

   # A vector of column names to be written for each pattern file line
   PatternFileColumns = cms.vstring(),
   # A vector of the lengths (in bits!) of each column
   PatternFileLengths = cms.vuint32(),
   # A vector of default values for each column
   PatternFileDefaultValues = cms.vuint32(),

   # By default, do not add comments with detailed information
   DebugOutput = cms.bool(False)
)

def addBlock(analyzer, name, count, length, default):
    for i in range(1,count+1):
        analyzer.PatternFileColumns.append("%s%d" % (name, i))
        analyzer.PatternFileLengths.append(length)
        analyzer.PatternFileDefaultValues.append(default)

def addPSB(analyzer, name):
    addBlock(analyzer, name, 4, 16, 0)


def addRegionalMuons(analyzer, name):
    # regional muons are different - they need to have a default of 0x0000ff00 when
    # empty to make input cable disconnects recognizable
    addBlock(analyzer, name, 4, 32, 0x0000ff00)

def addGMTMuons(analyzer, name):
    addBlock(analyzer, name, 4, 26, 0)

# set up format:
fields   = l1GtPatternGenerator.PatternFileColumns
lengths  = l1GtPatternGenerator.PatternFileLengths
defaults = l1GtPatternGenerator.PatternFileDefaultValues

# column 1..20: some fairly standard PSBs (calo1 - calo5)
for name in [ "gctIsoEm", "gctEm", "cenJet", "forJet", "tauJet" ]:
    addPSB(l1GtPatternGenerator, name)

# then the energy sums, which are slightly more complicated
# (calo6)
fields   += ["etTotal1", "etMiss1", "etHad1", "etMissPhi1"]
lengths  += [        16,       16,        16,           16]
defaults += [        0,       0,        0,           0]

# HF bit counts / etsums (which are mangled in the C++ code)
# (calo7)
fields   += [ "hfPsbValue1_l", "htMiss1", "hfPsbValue1_h", "unknown"]
lengths  += [             16,        16,             16,         16]
defaults += [              0,         0,              0,          0]

# calo8 - free
addPSB(l1GtPatternGenerator, "unknown")

# calo9 - "totem", currently
addPSB(l1GtPatternGenerator, "unknown")

# calo 10 
# BPTX/Castor and TBD data - default to 0xffff to get BPTX triggers matching GT emulator
addBlock(l1GtPatternGenerator, "unknown", 4, 16, 0xffff)

# 12 more PSBs we don't fill
for i in range(12):
    addPSB(l1GtPatternGenerator, "unknown")

# regional muons
addRegionalMuons(l1GtPatternGenerator, "fwdMuon")
addRegionalMuons(l1GtPatternGenerator, "cscMuon")
addRegionalMuons(l1GtPatternGenerator, "dtMuon")
addRegionalMuons(l1GtPatternGenerator, "brlMuon")

# global muons
addGMTMuons(l1GtPatternGenerator, "gmtMuon")

# GT stuff
addBlock(l1GtPatternGenerator, "gtDecisionExt", 2, 32, 0)
addBlock(l1GtPatternGenerator, "gtDecision", 4, 32, 0)

# tech triggers: a bit complicated, since we like to mix up
#                half-words (see header)
fields   += ["gtTechTrigger1_l", "gtTechTrigger2_l", "gtTechTrigger1_h", "gtTechTrigger2_h"]
lengths  += [                16,                 16,                 16,                 16]
defaults += [                 0,                  0,                  0,                  0]

fields   += ["gtFinalOr"]
lengths  += [          9]
defaults += [          0]

# just to make sure the python magic adds up to the proper output format
if len(fields) != 119:
    raise ValueError("Expecting 119 data fields (120 - event number) in pattern file format, got %d!" % len(fields) )

# For debugging: Get an overview of your pattern file format
#print fields
#print lengths
#print defaults

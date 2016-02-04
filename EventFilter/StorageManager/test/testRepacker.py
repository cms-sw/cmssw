
# Define a job to test some features of a repacker executable.
# Note that this is in many ways trivial compared to what the
# final repacker executable will need to be.

# This reads a single streamer file as input and then outputs
# 50 files using 50 PoolOutputModules.  Each output module
# selects events based on a bit in the triggerResults object
# in the input file.

# Original Author   W. David Dagenhart    19 June 2007

# This file started as a simple conversion of testRepacker.cfg
# from the old configuration language to PYTHON.

import FWCore.ParameterSet.Config as cms

process = cms.Process("REPACKER")

process.include("FWCore/MessageLogger/data/MessageLogger.cfi")

process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring( "ProductNotFound", "TooManyProducts", "TooFewProducts" )
)

process.source = cms.Source("NewEventStreamFileReader",
    fileNames = cms.untracked.vstring( 'file:testRandomFilter.dat' ),
    max_event_size = cms.int32(7000000),
    max_queue_depth = cms.int32(5)
)

process.out1 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( "p1.root" ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring( "p1:HLT" )
    )
)

process.out2 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( "p2.root" ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring( "p2:HLT" )
    )
)

process.out3 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( "p3.root" ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring( "p3:HLT" )
    )
)

process.out4 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( "p4.root" ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring( "p4:HLT" )
    )
)

process.out5 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( "p5.root" ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring( "p5:HLT" )
    )
)

process.out6 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( "p6.root" ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring( "p6:HLT" )
    )
)

process.out7 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( "p7.root" ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring( "p7:HLT" )
    )
)

process.out8 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( "p8.root" ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring( "p8:HLT" )
    )
)

process.out9 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( "p9.root" ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring( "p9:HLT" )
    )
)

process.out10 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( "p10.root" ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring( "p10:HLT" )
    )
)

process.out11 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( "p11.root" ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring( "p11:HLT" )
    )
)

process.out12 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( "p12.root" ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring( "p12:HLT" )
    )
)

process.out13 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( "p13.root" ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring( "p13:HLT" )
    )
)

process.out14 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( "p14.root" ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring( "p14:HLT" )
    )
)

process.out15 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( "p15.root" ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring( "p15:HLT" )
    )
)

process.out16 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( "p16.root" ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring( "p16:HLT" )
    )
)

process.out17 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( "p17.root" ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring( "p17:HLT" )
    )
)

process.out18 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( "p18.root" ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring( "p18:HLT" )
    )
)

process.out19 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( "p19.root" ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring( "p19:HLT" )
    )
)

process.out20 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( "p20.root" ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring( "p20:HLT" )
    )
)

process.out21 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( "p21.root" ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring( "p21:HLT" )
    )
)

process.out22 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( "p22.root" ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring( "p22:HLT" )
    )
)

process.out23 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( "p23.root" ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring( "p23:HLT" )
    )
)

process.out24 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( "p24.root" ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring( "p24:HLT" )
    )
)

process.out25 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( "p25.root" ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring( "p25:HLT" )
    )
)

process.out26 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( "p26.root" ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring( "p26:HLT" )
    )
)

process.out27 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( "p27.root" ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring( "p27:HLT" )
    )
)

process.out28 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( "p28.root" ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring( "p28:HLT" )
    )
)

process.out29 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( "p29.root" ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring( "p29:HLT" )
    )
)

process.out30 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( "p30.root" ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring( "p30:HLT" )
    )
)

process.out31 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( "p31.root" ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring( "p31:HLT" )
    )
)

process.out32 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( "p32.root" ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring( "p32:HLT" )
    )
)

process.out33 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( "p33.root" ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring( "p33:HLT" )
    )
)

process.out34 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( "p34.root" ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring( "p34:HLT" )
    )
)

process.out35 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( "p35.root" ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring( "p35:HLT" )
    )
)

process.out36 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( "p36.root" ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring( "p36:HLT" )
    )
)

process.out37 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( "p37.root" ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring( "p37:HLT" )
    )
)

process.out38 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( "p38.root" ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring( "p38:HLT" )
    )
)

process.out39 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( "p39.root" ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring( "p39:HLT" )
    )
)

process.out40 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( "p40.root" ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring( "p40:HLT" )
    )
)

process.out41 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( "p41.root" ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring( "p41:HLT" )
    )
)

process.out42 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( "p42.root" ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring( "p42:HLT" )
    )
)

process.out43 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( "p43.root" ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring( "p43:HLT" )
    )
)

process.out44 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( "p44.root" ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring( "p44:HLT" )
    )
)

process.out45 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( "p45.root" ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring( "p45:HLT" )
    )
)

process.out46 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( "p46.root" ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring( "p46:HLT" )
    )
)

process.out47 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( "p47.root" ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring( "p47:HLT" )
    )
)

process.out48 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( "p48.root" ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring( "p48:HLT" )
    )
)

process.out49 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( "p49.root" ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring( "p49:HLT" )
    )
)

process.out50 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( "p50.root" ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring( "p50:HLT" )
    )
)

process.o = cms.EndPath( process.out1 +
                         process.out2 +
                         process.out3 +
                         process.out4 +
                         process.out5 +
                         process.out6 +
                         process.out7 +
                         process.out8 +
                         process.out9 +
                         process.out10 +
                         process.out11 +
                         process.out12 +
                         process.out13 +
                         process.out14 +
                         process.out15 +
                         process.out16 +
                         process.out17 +
                         process.out18 +
                         process.out19 +
                         process.out20 +
                         process.out21 +
                         process.out22 +
                         process.out23 +
                         process.out24 +
                         process.out25 +
                         process.out26 +
                         process.out27 +
                         process.out28 +
                         process.out29 +
                         process.out30 +
                         process.out31 +
                         process.out32 +
                         process.out33 +
                         process.out34 +
                         process.out35 +
                         process.out36 +
                         process.out37 +
                         process.out38 +
                         process.out39 +
                         process.out40 +
                         process.out41 +
                         process.out42 +
                         process.out43 +
                         process.out44 +
                         process.out45 +
                         process.out46 +
                         process.out47 +
                         process.out48 +
                         process.out49 +
                         process.out50
)

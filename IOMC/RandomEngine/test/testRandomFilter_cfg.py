# Define a job that produces some fake data useful for
# repacker testing. The output is in Streamer format.
# Original Author   W. David Dagenhart    27 March 2007

import FWCore.ParameterSet.Config as cms

process = cms.Process("HLT")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

process.source = cms.Source("EmptySource")

# ~3.0 GB output file = 1.5 MB events x 2290 events x 0.87,
# where 0.87 is the compression factor empirically measured
# for this fake data.  There is a problems with this.
# On machines that are not the newest, the streamer
# output module crashes when the file size goes over
# 2 GB. We believe this is related to the system libraries
# used for file I/O.  On a new sl4 64 bit machine and compiling
# with -m32 option, the streamer output module runs to
# completion and creates a 3 GB without crashing.     

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2290)
# ~2.0 GB output file = 1.5 MB per event x 1270 events
# The actual output file size is closer to 1.7 GB because
# of compression.
#    input = cms.untracked.int32(1270)
)

# This produces fake data to fill the events.  It is just
# vectors full of random numbers.
# ~1.5 MB event = 4 bytes x 2500 vector elements x 150 vectors.
# Produces 150 branches, which is of order what we expect in data.
# The bit mask is 2^24 - 1, this is used to set the top 8 bits
# of each array element to zero.  Without use of this bit mask,
# the bits are all randomly set.  The reason we care about this
# is that we want the data to be smaller after compression
# is run, because it is more like real data.
process.prod = cms.EDProducer("StreamThingProducer",
    array_size = cms.int32(2500),
    instance_count = cms.int32(150),
    apply_bit_mask = cms.untracked.bool(True),
    bit_mask = cms.untracked.uint32(16777215)
)

# Set up 50 filter modules, each with its own path.  These
# make filter decisions based on 50 independent random
# number sequences and an accept rate configured below.
# The TriggerResults object created in the event will store
# the results of these filter modules.

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    m1 = cms.PSet(
        initialSeed = cms.untracked.uint32(1)
    ),
    m2 = cms.PSet(
        initialSeed = cms.untracked.uint32(2)
    ),
    m3 = cms.PSet(
        initialSeed = cms.untracked.uint32(3)
    ),
    m4 = cms.PSet(
        initialSeed = cms.untracked.uint32(4)
    ),
    m5 = cms.PSet(
        initialSeed = cms.untracked.uint32(5)
    ),
    m6 = cms.PSet(
        initialSeed = cms.untracked.uint32(6)
    ),
    m7 = cms.PSet(
        initialSeed = cms.untracked.uint32(7)
    ),
    m8 = cms.PSet(
        initialSeed = cms.untracked.uint32(8)
    ),
    m9 = cms.PSet(
        initialSeed = cms.untracked.uint32(9)
    ),
    m10 = cms.PSet(
        initialSeed = cms.untracked.uint32(10)
    ),
    m11 = cms.PSet(
        initialSeed = cms.untracked.uint32(11)
    ),
    m12 = cms.PSet(
        initialSeed = cms.untracked.uint32(12)
    ),
    m13 = cms.PSet(
        initialSeed = cms.untracked.uint32(13)
    ),
    m14 = cms.PSet(
        initialSeed = cms.untracked.uint32(14)
    ),
    m15 = cms.PSet(
        initialSeed = cms.untracked.uint32(15)
    ),
    m16 = cms.PSet(
        initialSeed = cms.untracked.uint32(16)
    ),
    m17 = cms.PSet(
        initialSeed = cms.untracked.uint32(17)
    ),
    m18 = cms.PSet(
        initialSeed = cms.untracked.uint32(18)
    ),
    m19 = cms.PSet(
        initialSeed = cms.untracked.uint32(19)
    ),
    m20 = cms.PSet(
        initialSeed = cms.untracked.uint32(20)
    ),
    m21 = cms.PSet(
        initialSeed = cms.untracked.uint32(21)
    ),
    m22 = cms.PSet(
        initialSeed = cms.untracked.uint32(22)
    ),
    m23 = cms.PSet(
        initialSeed = cms.untracked.uint32(23)
    ),
    m24 = cms.PSet(
        initialSeed = cms.untracked.uint32(24)
    ),
    m25 = cms.PSet(
        initialSeed = cms.untracked.uint32(25)
    ),
    m26 = cms.PSet(
        initialSeed = cms.untracked.uint32(26)
    ),
    m27 = cms.PSet(
        initialSeed = cms.untracked.uint32(27)
    ),
    m28 = cms.PSet(
        initialSeed = cms.untracked.uint32(28)
    ),
    m29 = cms.PSet(
        initialSeed = cms.untracked.uint32(29)
    ),
    m30 = cms.PSet(
        initialSeed = cms.untracked.uint32(30)
    ),
    m31 = cms.PSet(
        initialSeed = cms.untracked.uint32(31)
    ),
    m32 = cms.PSet(
        initialSeed = cms.untracked.uint32(32)
    ),
    m33 = cms.PSet(
        initialSeed = cms.untracked.uint32(33)
    ),
    m34 = cms.PSet(
        initialSeed = cms.untracked.uint32(34)
    ),
    m35 = cms.PSet(
        initialSeed = cms.untracked.uint32(35)
    ),
    m36 = cms.PSet(
        initialSeed = cms.untracked.uint32(36)
    ),
    m37 = cms.PSet(
        initialSeed = cms.untracked.uint32(37)
    ),
    m38 = cms.PSet(
        initialSeed = cms.untracked.uint32(38)
    ),
    m39 = cms.PSet(
        initialSeed = cms.untracked.uint32(39)
    ),
    m40 = cms.PSet(
        initialSeed = cms.untracked.uint32(40)
    ),
    m41 = cms.PSet(
        initialSeed = cms.untracked.uint32(41)
    ),
    m42 = cms.PSet(
        initialSeed = cms.untracked.uint32(42)
    ),
    m43 = cms.PSet(
        initialSeed = cms.untracked.uint32(43)
    ),
    m44 = cms.PSet(
        initialSeed = cms.untracked.uint32(44)
    ),
    m45 = cms.PSet(
        initialSeed = cms.untracked.uint32(45)
    ),
    m46 = cms.PSet(
        initialSeed = cms.untracked.uint32(46)
    ),
    m47 = cms.PSet(
        initialSeed = cms.untracked.uint32(47)
    ),
    m48 = cms.PSet(
        initialSeed = cms.untracked.uint32(48)
    ),
    m49 = cms.PSet(
        initialSeed = cms.untracked.uint32(49)
    ),
    m50 = cms.PSet(
        initialSeed = cms.untracked.uint32(50)
    )
)

process.m1 = cms.EDFilter("RandomFilter",
    acceptRate = cms.untracked.double(0.2)
)

process.m2 = cms.EDFilter("RandomFilter",
    acceptRate = cms.untracked.double(0.2)
)

process.m3 = cms.EDFilter("RandomFilter",
    acceptRate = cms.untracked.double(0.2)
)

process.m4 = cms.EDFilter("RandomFilter",
    acceptRate = cms.untracked.double(0.2)
)

process.m5 = cms.EDFilter("RandomFilter",
    acceptRate = cms.untracked.double(0.2)
)

process.m6 = cms.EDFilter("RandomFilter",
    acceptRate = cms.untracked.double(0.1)
)

process.m7 = cms.EDFilter("RandomFilter",
    acceptRate = cms.untracked.double(0.1)
)

process.m8 = cms.EDFilter("RandomFilter",
    acceptRate = cms.untracked.double(0.1)
)

process.m9 = cms.EDFilter("RandomFilter",
    acceptRate = cms.untracked.double(0.1)
)

process.m10 = cms.EDFilter("RandomFilter",
    acceptRate = cms.untracked.double(0.1)
)

process.m11 = cms.EDFilter("RandomFilter",
    acceptRate = cms.untracked.double(0.05)
)

process.m12 = cms.EDFilter("RandomFilter",
    acceptRate = cms.untracked.double(0.05)
)

process.m13 = cms.EDFilter("RandomFilter",
    acceptRate = cms.untracked.double(0.05)
)

process.m14 = cms.EDFilter("RandomFilter",
    acceptRate = cms.untracked.double(0.05)
)

process.m15 = cms.EDFilter("RandomFilter",
    acceptRate = cms.untracked.double(0.05)
)

process.m16 = cms.EDFilter("RandomFilter",
    acceptRate = cms.untracked.double(0.05)
)

process.m17 = cms.EDFilter("RandomFilter",
    acceptRate = cms.untracked.double(0.05)
)

process.m18 = cms.EDFilter("RandomFilter",
    acceptRate = cms.untracked.double(0.05)
)

process.m19 = cms.EDFilter("RandomFilter",
    acceptRate = cms.untracked.double(0.05)
)

process.m20 = cms.EDFilter("RandomFilter",
    acceptRate = cms.untracked.double(0.05)
)

process.m21 = cms.EDFilter("RandomFilter",
    acceptRate = cms.untracked.double(0.01)
)

process.m22 = cms.EDFilter("RandomFilter",
    acceptRate = cms.untracked.double(0.01)
)

process.m23 = cms.EDFilter("RandomFilter",
    acceptRate = cms.untracked.double(0.01)
)

process.m24 = cms.EDFilter("RandomFilter",
    acceptRate = cms.untracked.double(0.01)
)

process.m25 = cms.EDFilter("RandomFilter",
    acceptRate = cms.untracked.double(0.01)
)

process.m26 = cms.EDFilter("RandomFilter",
    acceptRate = cms.untracked.double(0.01)
)

process.m27 = cms.EDFilter("RandomFilter",
    acceptRate = cms.untracked.double(0.01)
)

process.m28 = cms.EDFilter("RandomFilter",
    acceptRate = cms.untracked.double(0.01)
)

process.m29 = cms.EDFilter("RandomFilter",
    acceptRate = cms.untracked.double(0.01)
)

process.m30 = cms.EDFilter("RandomFilter",
    acceptRate = cms.untracked.double(0.01)
)

process.m31 = cms.EDFilter("RandomFilter",
    acceptRate = cms.untracked.double(0.005)
)

process.m32 = cms.EDFilter("RandomFilter",
    acceptRate = cms.untracked.double(0.005)
)

process.m33 = cms.EDFilter("RandomFilter",
    acceptRate = cms.untracked.double(0.005)
)

process.m34 = cms.EDFilter("RandomFilter",
    acceptRate = cms.untracked.double(0.005)
)

process.m35 = cms.EDFilter("RandomFilter",
    acceptRate = cms.untracked.double(0.005)
)

process.m36 = cms.EDFilter("RandomFilter",
    acceptRate = cms.untracked.double(0.005)
)

process.m37 = cms.EDFilter("RandomFilter",
    acceptRate = cms.untracked.double(0.005)
)

process.m38 = cms.EDFilter("RandomFilter",
    acceptRate = cms.untracked.double(0.005)
)

process.m39 = cms.EDFilter("RandomFilter",
    acceptRate = cms.untracked.double(0.005)
)

process.m40 = cms.EDFilter("RandomFilter",
    acceptRate = cms.untracked.double(0.005)
)

process.m41 = cms.EDFilter("RandomFilter",
    acceptRate = cms.untracked.double(0.001)
)

process.m42 = cms.EDFilter("RandomFilter",
    acceptRate = cms.untracked.double(0.001)
)

process.m43 = cms.EDFilter("RandomFilter",
    acceptRate = cms.untracked.double(0.001)
)

process.m44 = cms.EDFilter("RandomFilter",
    acceptRate = cms.untracked.double(0.001)
)

process.m45 = cms.EDFilter("RandomFilter",
    acceptRate = cms.untracked.double(0.001)
)

process.m46 = cms.EDFilter("RandomFilter",
    acceptRate = cms.untracked.double(0.0001)
)

process.m47 = cms.EDFilter("RandomFilter",
    acceptRate = cms.untracked.double(0.0001)
)

process.m48 = cms.EDFilter("RandomFilter",
    acceptRate = cms.untracked.double(0.0001)
)

process.m49 = cms.EDFilter("RandomFilter",
    acceptRate = cms.untracked.double(0.0001)
)

process.m50 = cms.EDFilter("RandomFilter",
    acceptRate = cms.untracked.double(0.0001)
)

process.out = cms.OutputModule("EventStreamFileWriter",
    max_event_size = cms.untracked.int32(7000000),
    use_compression = cms.untracked.bool(True),
    compression_level = cms.untracked.int32(1),
    fileName = cms.untracked.string('testRandomFilter.dat'),
)

#  processs.outp = cms.OutputModule("PoolOutputModule",
#      fileName = cms.untracked.string.("testRandomFilter.root")
#  )

process.makeData = cms.Path(process.prod)
process.p1 = cms.Path(process.m1)
process.p2 = cms.Path(process.m2)
process.p3 = cms.Path(process.m3)
process.p4 = cms.Path(process.m4)
process.p5 = cms.Path(process.m5)
process.p6 = cms.Path(process.m6)
process.p7 = cms.Path(process.m7)
process.p8 = cms.Path(process.m8)
process.p9 = cms.Path(process.m9)
process.p10 = cms.Path(process.m10)
process.p11 = cms.Path(process.m11)
process.p12 = cms.Path(process.m12)
process.p13 = cms.Path(process.m13)
process.p14 = cms.Path(process.m14)
process.p15 = cms.Path(process.m15)
process.p16 = cms.Path(process.m16)
process.p17 = cms.Path(process.m17)
process.p18 = cms.Path(process.m18)
process.p19 = cms.Path(process.m19)
process.p20 = cms.Path(process.m20)
process.p21 = cms.Path(process.m21)
process.p22 = cms.Path(process.m22)
process.p23 = cms.Path(process.m23)
process.p24 = cms.Path(process.m24)
process.p25 = cms.Path(process.m25)
process.p26 = cms.Path(process.m26)
process.p27 = cms.Path(process.m27)
process.p28 = cms.Path(process.m28)
process.p29 = cms.Path(process.m29)
process.p30 = cms.Path(process.m30)
process.p31 = cms.Path(process.m31)
process.p32 = cms.Path(process.m32)
process.p33 = cms.Path(process.m33)
process.p34 = cms.Path(process.m34)
process.p35 = cms.Path(process.m35)
process.p36 = cms.Path(process.m36)
process.p37 = cms.Path(process.m37)
process.p38 = cms.Path(process.m38)
process.p39 = cms.Path(process.m39)
process.p40 = cms.Path(process.m40)
process.p41 = cms.Path(process.m41)
process.p42 = cms.Path(process.m42)
process.p43 = cms.Path(process.m43)
process.p44 = cms.Path(process.m44)
process.p45 = cms.Path(process.m45)
process.p46 = cms.Path(process.m46)
process.p47 = cms.Path(process.m47)
process.p48 = cms.Path(process.m48)
process.p49 = cms.Path(process.m49)
process.p50 = cms.Path(process.m50)
process.o = cms.EndPath(process.out)

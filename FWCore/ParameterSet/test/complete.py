#------------------------------------------------------------
#
# This is an annotated example of a configuration file.
# It is intended to illustrate the mapping between the configuration
# language and the Python representation of the configuration
# description that is used by the Data Management tools.
#
# $Id: complete.py,v 1.5 2007/08/31 21:32:49 rpw Exp $
#
#------------------------------------------------------------


# The configuration grammar accepts documents which have the general
# form of a nested collection of 'blocks' of various kinds. These
# blocks are delimted by braces: '{' and '}'.

# Newlines are meaningless in the configuration langauge.

# Comments (obviously!) are introduced by a sharp ('#'), and continue
# to the end of line.

# C++-style comments (beginning with a cms.double(d forward slash: '//')
# are also accepted.

# Each configuration must have exactly one process block.

# Eventually, the software may restrict the process name to being one
# of a recognized set (such as HLT, PROD, TEST, USER). This is not yet
# implemented.
import FWCore.ParameterSet.Config as cms

process = cms.Process("PROCESSNAME")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring("file:no_real_file_here"),
    debugFlag = cms.untracked.bool(False),
    maxEvents = cms.untracked.int32(-1),
    debugVebosity = cms.untracked.uint32(10)
  )

process.a = cms.EDProducer("AProducer",
    a = cms.int32(32),
    h1 = cms.uint32(0xCFDFEFFF),
    b = cms.vdouble(1.1, 2.2),
    c = cms.vstring(),
    d = cms.vstring('boo', "yah" ),
    nothing =cms.string(""),
    moreNothing = cms.string(''),
    absolutelyNothing = cms.string('\0'),
    justATick = cms.string('\'')
  )

process.b = cms.EDProducer("BProducer",
   a = cms.untracked.int32(14),
   b= cms.string("sillyness ensues"),
   c = cms.PSet
   (
     a = cms.string('nested'),
     b = cms.string("more")
   ),
   d = cms.VPSet(cms.PSet(i=cms.int32(10101),
                  b = cms.bool(False) ), cms.PSet() ),
   e = cms.VPSet(),
   f = cms.VPSet(cms.PSet(inner = cms.VPSet()), cms.PSet() ),
   tag = cms.InputTag("y","z"),
   tags = cms.VInputTag(cms.InputTag("a","b"), cms.InputTag("c"), cms.InputTag("d","e"))
  )


process.c = cms.EDProducer("CProducer")

process.y = cms.EDProducer("PoolOutputModule",
  fileName = cms.untracked.string("file:myfile_y.root"),
  maxEvents = cms.untracked.int32(2112)
 )


process.z = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string("file:myfile_z.root"),
    maxEvents = cms.untracked.int32(91624)
  )

process.vp = cms.EDProducer("VPProducer",
    c = cms.PSet(
      d = cms.VPSet(
          cms.PSet(
            a = cms.int32(32),
            d = cms.PSet(
              e = cms.VPSet()
            )
          ),
          cms.PSet(b=cms.int32(12))
      )
    )
  )

process.s1 = cms.Sequence(process.a+process.b)
process.s2 = cms.Sequence(process.b)
process.s3 = cms.Sequence(process.a)
  # It is not an error for two sequences (here, s3 and s4) to be identical.
process.s4 = cms.Sequence(process.a)

process.p1 = cms.Path((process.a+process.b)* process.c )
process.p2 = cms.Path(process.s1+ (process.s3*process.s2) )
  
process.ep1 = cms.EndPath(process.y*process.z)

process.schedule = cms.Schedule(process.p1, process.p2, process.ep1)

process.ess1 = cms.ESSource("ESSType1",
    b=cms.int32(2)
  )

ESSType1 = cms.ESSource("ESSType1",
    b=cms.int32(0)
  )
process.add_(ESSType1)

ESSType2 = cms.ESSource("ESSType2",
   x = cms.double(1.5)
  )
process.add_(ESSType2)

process.esm1 = cms.ESProducer("ESMType1",
    s= cms.string("hi")
  )

ESMType2 = cms.ESProducer("ESMType2", 
    a=cms.int32(3)
  )
process.add_(ESMType2)

s1 = cms.Service("ServiceType1", 
  b=cms.double(2)
)

s2 = cms.Service("ServiceType2", 
  x=cms.double(1.5)
)

process.add_(s1)
process.add_(s2)

process.looper = cms.Looper("ALooper",
  nLoops = cms.uint32(10)
)

process.mix = cms.EDProducer("MixingModule",
    input = cms.SecSource("PoolSource",
      fileNames = cms.untracked.vstring("file:pileup.root")
    ),
    mixtype = cms.string("fixed"),
    average_number = cms.double(14.3),
    min_bunch = cms.int32(-5),
    max_bunch = cms.int32(3)
  )



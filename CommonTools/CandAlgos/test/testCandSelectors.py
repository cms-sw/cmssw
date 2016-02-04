from FWCore.ParameterSet.Config import *

process = Process("CandSelectorTest")

process.include("FWCore/MessageLogger/data/MessageLogger.cfi")

process.maxEvents = untracked.PSet( input = untracked.int32(10) )

process.source = Source("PoolSource",
  fileNames = untracked.vstring("file:genevents.root")
)

# select the 10 particles with the larget Pt
process.largestPtCands = EDProducer("LargestPtCandSelector",
  src = InputTag("genParticleCandidates"),
  maxNumber = uint32( 10 )
)

# select only electrons, and save a vector of references 
process.electronRefs = EDProducer("PdgIdCandRefVectorSelector",
  src = InputTag("genParticleCandidates"),
  pdgId = vint32( 11 )
)

# select only electrons, and save clones
process.electrons = EDProducer("PdgIdCandSelector",
  src = InputTag("genParticleCandidates"),
  pdgId = vint32( 11 )
)

# select only muons, and save a vector of references 
process.muonRefs = EDProducer("PdgIdCandRefVectorSelector",
  src = InputTag("genParticleCandidates"),
  pdgId = vint32( 13 )
)

# select only muons, and save clones
process.muons = EDProducer("PdgIdCandSelector",
  src = InputTag("genParticleCandidates"),
  pdgId = vint32( 13 )
)

# select only electrons within eta and Pt cuts 
process.bestElectrons = EDFilter("EtaPtMinCandViewSelector",
  src = InputTag("electronRefs"),
  ptMin = double( 20 ),
  etaMin = double( -2.5 ),
  etaMax = double( 2.5 )
)

# make Z->e+e-
process.zCands = EDProducer("CandShallowCloneCombiner",
  decay = string("electrons@+ electrons@-"),
  cut = string("20 < mass < 200")
)

# make exotic decay to three electron
process.exoticCands = EDProducer("CandShallowCloneCombiner",
  decay = string("electrons@+ electrons@- electrons@+"),
  cut = string("20 < mass < 400")
)

# merge muons and electrons into leptons
process.leptons = EDProducer("CandMerger",
  src = VInputTag("electrons", "muons")
)

process.out = OutputModule("PoolOutputModule",
  fileName = untracked.string("cands.root")
)

process.printEventNumber = OutputModule("AsciiOutputModule")

process.select = Path(
  process.largestPtCands *
  process.electronRefs *
  process.electrons *
  process.muonRefs *
  process.muons *
  process.bestElectrons *
  process.leptons *
  process.zCands *
  process.exoticCands
)
 	
process.ep = EndPath(
  process.printEventNumber *
  process.out
)

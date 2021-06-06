import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("BeamBackground")

options = VarParsing.VarParsing("analysis")

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')

options.register ('globalTag',
                  "DONOTEXIST",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "GlobalTag")

options.parseArguments()


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(500)
)


process.source = cms.Source("PoolSource",
                    fileNames = cms.untracked.vstring(options.inputFiles),
                    secondaryFileNames = cms.untracked.vstring()
)


from HLTrigger.HLTfilters.triggerResultsFilter_cfi import *
process.hltSelection = triggerResultsFilter.clone(
                                          triggerConditions = cms.vstring("HLT_ZeroBias_*"),
                                          hltResults = cms.InputTag( "TriggerResults", "", "HLT" ),
                                          l1tResults = cms.InputTag( "" ),
                                          throw = cms.bool(False)
                                          )


process.load("DPGAnalysis.SiStripTools.sipixelclustermultiplicityprod_cfi")
process.load("DPGAnalysis.SiStripTools.sistripclustermultiplicityprod_cfi")

ssclustermultprod = cms.EDProducer("SiStripClusterMultiplicityProducer",
                                   clusterdigiCollection = cms.InputTag("siStripClusters"),
                                   wantedSubDets = cms.VPSet(    
                                                          cms.PSet(detSelection = cms.uint32(0),detLabel = cms.string("TK")),
                                                          cms.PSet(detSelection = cms.uint32(3),detLabel = cms.string("TIB")),
                                                          cms.PSet(detSelection = cms.uint32(4),detLabel = cms.string("TID")),
                                                          cms.PSet(detSelection = cms.uint32(5),detLabel = cms.string("TOB")),
                                                          cms.PSet(detSelection = cms.uint32(6),detLabel = cms.string("TEC"))
                                                          )
                                )


spclustermultprod = cms.EDProducer("SiPixelClusterMultiplicityProducer",
                                   clusterdigiCollection = cms.InputTag("siPixelClusters"),
                                   wantedSubDets = cms.VPSet(    
                                                          cms.PSet(detSelection = cms.uint32(0),detLabel = cms.string("Pixel")),
                                                          cms.PSet(detSelection = cms.uint32(1),detLabel = cms.string("BPIX")),
                                                          cms.PSet(detSelection = cms.uint32(2),detLabel = cms.string("FPIX"))
                                                          )
                                )


process.load("DPGAnalysis.SiStripTools.multiplicitycorr_cfi")

process.multiplicitycorr.correlationConfigurations = cms.VPSet(
   cms.PSet(xMultiplicityMap = cms.InputTag("ssclustermultprod"),
            xDetSelection = cms.uint32(0), xDetLabel = cms.string("TK"), xBins = cms.uint32(3000), xMax=cms.double(100000), 
            yMultiplicityMap = cms.InputTag("spclustermultprod"),
            yDetSelection = cms.uint32(0), yDetLabel = cms.string("Pixel"), yBins = cms.uint32(1000), yMax=cms.double(30000),
            rBins = cms.uint32(200), scaleFactor = cms.untracked.double(10.4),#10.4 for 25ns //  7.7 for 50 ns
            runHisto=cms.bool(True),runHistoBXProfile=cms.bool(True),runHistoBX=cms.bool(True),runHisto2D=cms.bool(True))
)


process.load("DPGAnalysis.SiStripTools.eventtimedistribution_cfi")


import DPGAnalysis.SiStripTools.apvcyclephaseproducerfroml1tsDB_cfi 
process.APVPhases = DPGAnalysis.SiStripTools.apvcyclephaseproducerfroml1tsDB_cfi.APVPhases 

process.load("DPGAnalysis.SiStripTools.eventwithhistoryproducerfroml1abc_cfi")

process.load("DPGAnalysis.SiStripTools.bysipixelvssistripclustmulteventfilter_cfi")

# mult2= Number of Strip Clusters; mult1= Number of Pixel Clusters

#filter for BeamBackground events
process.offdiagonal = process.bysipixelvssistripclustmulteventfilter.clone(cut=cms.string("( mult2 < 7.4*(mult1-300))")) #7.4 for 25ns // 5.5 for 50 ns
process.NoZeroSClusters = process.bysipixelvssistripclustmulteventfilter.clone(cut=cms.string("( mult2 > 500)"))

#Filter for high strip noise
#process.offdiagonal = process.bysipixelvssistripclustmulteventfilter.clone(cut=cms.string("( mult2 > 7.4*(mult1+300))")) #7.4 for 25ns // 5.5 for 50 ns
#process.NoZeroSClusters = process.bysipixelvssistripclustmulteventfilter.clone(cut=cms.string("( mult2 > 26000)"))

#Filter for strip noise
#process.offdiagonal = process.bysipixelvssistripclustmulteventfilter.clone(cut=cms.string("( mult2 > 7.4*(mult1+300))")) #7.4 for 25ns // 5.5 for 50 ns
#process.NoZeroSClusters = process.bysipixelvssistripclustmulteventfilter.clone(cut=cms.string("( mult2 < 26000)"))

#Filter for main diagonal
#process.offdiagonal = process.bysipixelvssistripclustmulteventfilter.clone(cut=cms.string("( mult2 > 7.4*(mult1-300))")) #7.4 for 25ns // 5.5 for 50 ns
#process.offdiagonal1 = process.bysipixelvssistripclustmulteventfilter.clone(cut=cms.string("( mult2 < 7.4*(mult1+300))")) #7.4 for 25ns // 5.5 for 50 ns




process.multiplicitycorrAfter=process.multiplicitycorr.clone()

process.multiplicitycorrAfter.correlationConfigurations = cms.VPSet(
   cms.PSet(xMultiplicityMap = cms.InputTag("ssclustermultprod"),
            xDetSelection = cms.uint32(0), xDetLabel = cms.string("TK"), xBins = cms.uint32(3000), xMax=cms.double(100000), 
            yMultiplicityMap = cms.InputTag("spclustermultprod"),
            yDetSelection = cms.uint32(0), yDetLabel = cms.string("Pixel"), yBins = cms.uint32(1000), yMax=cms.double(30000),
            rBins = cms.uint32(200), scaleFactor = cms.untracked.double(10.4),#10.4 for 25 ns
            runHisto=cms.bool(True),runHistoBXProfile=cms.bool(True),runHistoBX=cms.bool(True),runHisto2D=cms.bool(True))
)

process.eventtimedistributionAfter= process.eventtimedistribution.clone()

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

from Configuration.AlCa.GlobalTag import GlobalTag

process.GlobalTag = GlobalTag(process.GlobalTag,options.globalTag, '')

process.TFileService = cms.Service('TFileService',
                                   fileName = cms.string('BeamBackground.root')
                                   )

process.MainSeq= cms.Sequence(process.hltSelection+process.consecutiveHEs+process.APVPhases+process.ssclustermultprod+process.spclustermultprod+process.eventtimedistribution+process.multiplicitycorr+process.NoZeroSClusters+process.offdiagonal+process.eventtimedistributionAfter+process.multiplicitycorrAfter)

process.p0 = cms.Path(process.MainSeq)

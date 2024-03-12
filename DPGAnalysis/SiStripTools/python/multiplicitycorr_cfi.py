import FWCore.ParameterSet.Config as cms

multiplicitycorr = cms.EDAnalyzer('MultiplicityCorrelator',
                            correlationConfigurations = cms.VPSet(    
    cms.PSet(xMultiplicityMap = cms.InputTag("ssclustermultprod"), xDetSelection = cms.uint32(0), xDetLabel = cms.string("TK"), xBins = cms.uint32(1000), xMax=cms.double(50000), 
             yMultiplicityMap = cms.InputTag("spclustermultprod"), yDetSelection = cms.uint32(0), yDetLabel = cms.string("Pixel"), yBins = cms.uint32(1000), yMax=cms.double(20000),
             rBins = cms.uint32(200), scaleFactor = cms.untracked.double(5.),
             runHisto=cms.bool(False),runHistoBXProfile=cms.bool(False),runHistoBX=cms.bool(False),runHisto2D=cms.bool(False)),
    cms.PSet(xMultiplicityMap = cms.InputTag("spclustermultprod"), xDetSelection = cms.uint32(0), xDetLabel = cms.string("Pixel"), xBins = cms.uint32(1000), xMax=cms.double(20000),
             yMultiplicityMap = cms.InputTag("spclustermultprod"), yDetSelection = cms.uint32(1), yDetLabel = cms.string("BPIX"), yBins = cms.uint32(1000), yMax=cms.double(20000),
             rBins = cms.uint32(200),
             runHisto=cms.bool(False),runHistoBXProfile=cms.bool(False),runHistoBX=cms.bool(False),runHisto2D=cms.bool(False))
    )
                                  )

# foo bar baz
# DpDqnchndGFJL

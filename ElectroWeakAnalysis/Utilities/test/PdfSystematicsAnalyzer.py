import FWCore.ParameterSet.Config as cms

# Process name
process = cms.Process("PDFANA")

# Max events
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
    #input = cms.untracked.int32(10)
)

# Printouts
process.MessageLogger = cms.Service("MessageLogger",
      cout = cms.untracked.PSet(
            default = cms.untracked.PSet(limit = cms.untracked.int32(100)),
            threshold = cms.untracked.string('INFO')
      ),
      destinations = cms.untracked.vstring('cout')
)

# Input files (on disk)
process.source = cms.Source("PoolSource",
      fileNames = cms.untracked.vstring("file:/data4/Wmunu_Summer09-MC_31X_V3_AODSIM-v1/0009/F82D4260-507F-DE11-B5D6-00093D128828.root")
)

# Produce PDF weights (maximum is 3)
process.pdfWeights = cms.EDProducer("PdfWeightProducer",
      PdfInfoTag = cms.untracked.InputTag("generator"),
      PdfSetNames = cms.untracked.vstring(
              "cteq65.LHgrid"
            #, "MRST2006nnlo.LHgrid"
            #, "MRST2007lomod.LHgrid"
      )
)

### NOTE: the following WMN selectors require the presence of
### the libraries and plugins fron the ElectroWeakAnalysis/WMuNu package
### So you need to process the ElectroWeakAnalysis/WMuNu package with
### some old CMSSW versions (at least <=3_1_2, <=3_3_0_pre4)
#

# Selector and parameters
# WMN fast selector (use W candidates in this example)
process.load("ElectroWeakAnalysis.WMuNu.WMuNuSelection_cff")

# Collect uncertainties for rate and acceptance
process.pdfSystematics = cms.EDFilter("PdfSystematicsAnalyzer",
      SelectorPath = cms.untracked.string('pdfana'),
      PdfWeightTags = cms.untracked.VInputTag(
              "pdfWeights:cteq65"
            #, "pdfWeights:MRST2006nnlo"
            #, "pdfWeights:MRST2007lomod"
      )
)

# Main path
process.pdfana = cms.Path(
       process.pdfWeights
      *process.selectCaloMetWMuNus
)

process.end = cms.EndPath(process.pdfSystematics)

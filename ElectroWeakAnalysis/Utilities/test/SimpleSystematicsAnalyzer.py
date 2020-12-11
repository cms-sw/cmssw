import FWCore.ParameterSet.Config as cms

# Process name
process = cms.Process("systAna")

# Max events
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
    #input = cms.untracked.int32(100)
)

# Printouts
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.enable = False
process.MessageLogger.cout = cms.untracked.PSet(
      enable = cms.untracked.bool(True),
      threshold = cms.untracked.string('INFO'),
      FwkReport = cms.untracked.PSet(reportEvery=cms.untracked.int32(100))
)

# Input files (on disk)
process.source = cms.Source("PoolSource",
      fileNames = cms.untracked.vstring("file:/ciet3a/data4/Wmunu_Summer09-MC_31X_V3_AODSIM-v1/0009/F82D4260-507F-DE11-B5D6-00093D128828.root")
      #fileNames = cms.untracked.vstring("file:/data4/ZMuMu_Summer09-MC_31X_V3_preproduction_312-v1_RECO/20A5B350-6979-DE11-A6EF-001560AD3140.root")
)

# Printout of generator information for the first event
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.printGenParticles = cms.EDAnalyzer("ParticleListDrawer",
  #maxEventsToPrint = cms.untracked.int32(-1),
  maxEventsToPrint = cms.untracked.int32(10),
  printVertex = cms.untracked.bool(False),
  src = cms.InputTag("genParticles")
)

### NOTE: the following WMN selectors require the presence of
### the libraries and plugins fron the ElectroWeakAnalysis/WMuNu package
### So you need to process the ElectroWeakAnalysis/WMuNu package with
### some old CMSSW versions (at least <=3_1_2, <=3_3_0_pre4)
#

# Selector and parameters
# WMN fast selector (use W candidates in this example)
process.load("ElectroWeakAnalysis.WMuNu.WMuNuSelection_cff")
# ZMM fast selector
#process.load("ElectroWeakAnalysis.Utilities.goldenZmmSelectionVBTF_cfi")

# Produce event weights according to generated boson Pt
# Example corresponds to approximate weights to study
# systematic effects due to ISR uncertainties (Z boson as fake example)
process.isrWeight = cms.EDProducer("ISRWeightProducer",
      GenTag = cms.untracked.InputTag("genParticles"),
      ISRBinEdges = cms.untracked.vdouble(
               0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.
            , 10., 11., 12., 13., 14., 15., 16., 17., 18., 19.
            , 20., 21., 22., 23., 24., 25., 26., 27., 28., 29.
            , 30., 31., 32., 33., 34., 35., 36., 37., 38., 39.
            , 40., 41., 42., 43., 44., 45., 46., 47., 48., 49.
            , 999999.
      ),
      PtWeights = cms.untracked.vdouble( 
              0.800665, 0.822121, 0.851249, 0.868285, 0.878733
            , 0.953853, 0.928108, 0.982021, 1.00659 , 1.00648
            , 1.03218 , 1.04924 , 1.03621 , 1.08743 , 1.01951
            , 1.10519 , 0.984263, 1.04853 , 1.06724 , 1.10183
            , 1.0503  , 1.13162 , 1.03837 , 1.12936 , 0.999173
            , 1.01453 , 1.11435 , 1.10545 , 1.07199 , 1.04542
            , 1.00828 , 1.0822  , 1.09667 , 1.16144 , 1.13906
            , 1.27974 , 1.14936 , 1.23235 , 1.06667 , 1.06363
            , 1.14225 , 1.22955 , 1.12674 , 1.03944 , 1.04639
            , 1.13667 , 1.20493 , 1.09349 , 1.2107  , 1.21073
      )
)

# Produce event weights to estimate missing QED FSR terms
process.fsrWeight = cms.EDProducer("FSRWeightProducer",
      GenTag = cms.untracked.InputTag("genParticles"),
)

# Produce event weights to estimate missing QED ISR terms
process.isrGammaWeight = cms.EDProducer("ISRGammaWeightProducer",
      GenTag = cms.untracked.InputTag("genParticles"),
)

# Produce weights for systematics
process.systematicsAnalyzer = cms.EDFilter("SimpleSystematicsAnalyzer",
      SelectorPath = cms.untracked.string('systAna'),
      WeightTags = cms.untracked.VInputTag("isrWeight","fsrWeight","isrGammaWeight")
)

# Main path
process.systAna = cms.Path(
       process.printGenParticles
      *process.isrWeight
      *process.fsrWeight
      *process.isrGammaWeight
      *process.selectCaloMetWMuNus
      #*process.goldenZMMSelectionSequence
)

process.end = cms.EndPath(process.systematicsAnalyzer)

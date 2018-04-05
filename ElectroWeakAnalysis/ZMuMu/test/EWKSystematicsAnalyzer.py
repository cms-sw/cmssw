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
process.MessageLogger.cerr.threshold = ''
process.MessageLogger.cerr.FwkReport.reportEvery = 1000


# Input files (on disk)
process.source = cms.Source("PoolSource",
      debugVerbosity = cms.untracked.uint32(0),
      debugFlag = cms.untracked.bool(False),
      fileNames = cms.untracked.vstring(
"file:~/Zmumu7TeVGenSimReco/0ABB0814-C082-DE11-9AB7-003048D4767C.root",
"file:~/Zmumu7TeVGenSimReco/0ABB0814-C082-DE11-9AB7-003048D4767C.root",
"file:~/Zmumu7TeVGenSimReco/38980FEC-C182-DE11-A3B5-003048D4767C.root",
 "file:~/Zmumu7TeVGenSimReco/3AF703B9-AE82-DE11-9656-0015172C0925.root",
"file:~/Zmumu7TeVGenSimReco/46854F8E-BC82-DE11-80AA-003048D47673.root",
 "file:~/Zmumu7TeVGenSimReco/8025F9B0-AC82-DE11-8C28-0015172560C6.root",
 "file:~/Zmumu7TeVGenSimReco/88DDF58E-BC82-DE11-ADD8-003048D47679.root",
 "file:~/Zmumu7TeVGenSimReco/9A115324-BB82-DE11-9C66-001517252130.root",
"file:~/Zmumu7TeVGenSimReco/FC279CAC-AD82-DE11-BAAA-001517357D36.root"
      )
)

# Printout of generator information for the first event
process.include("SimGeneral/HepPDTESSource/data/pythiapdt.cfi")
process.printGenParticles = cms.EDAnalyzer("ParticleListDrawer",
  maxEventsToPrint = cms.untracked.int32(10),
  printVertex = cms.untracked.bool(False),
  src = cms.InputTag("genParticles")
)


# Produce event weights according to generated boson Pt
# Example corresponds to approximate weights to study
# systematic effects due to ISR uncertainties (Z boson as fake example)
process.isrWeight = cms.EDProducer("ISRWeightProducer",
      GenTag = cms.untracked.InputTag("VtxSmeared"),
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

# Produce event weights to estimate missing O(alpha) terms + NLO QED terms
process.fsrWeight = cms.EDProducer("FSRWeightProducer",
      GenTag = cms.untracked.InputTag("VtxSmeared"),
)

# Produce event weights to estimate missing QED ISR terms
process.isrGammaWeight = cms.EDProducer("ISRGammaWeightProducer",
      GenTag = cms.untracked.InputTag("VtxSmeared"),
)

# Produce weights for systematics
process.systematicsAnalyzer = cms.EDFilter("SimpleSystematicsAnalyzer",
      SelectorPath = cms.untracked.string('systAna'),
      WeightTags = cms.untracked.VInputTag("isrWeight","fsrWeight","isrGammaWeight")
)


# Save weights in the output file 
process.load("Configuration.EventContent.EventContent_cff")
process.MyEventContent = cms.PSet( 
      outputCommands = process.AODSIMEventContent.outputCommands
)
process.MyEventContent.outputCommands.extend(
      cms.untracked.vstring('drop *',
                            'keep *_genParticles_*_*',
                            'keep *_isrWeight_*_*',
                            'keep *_fsrWeight_*_*',
                            'keep *_isrGammaWeight_*_*',
                           # 'keep *_MRST2007lomodewkPdfWeights_*_*',                                        'keep
# *_cteq6mLHewkPdfWeights_*_*',
 #                            'keep *_MRST2007lomodewkPdfWeights_*_*',                                        'kee
#p *_MRST2004nloewkPdfWeights_*_*',
 #                            'keep *_genEventWeight_*_*'
                            )
)

# Output (optionaly filtered by path)
process.Output = cms.OutputModule("PoolOutputModule",
    process.MyEventContent,
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('systAna')
    ),
    fileName = cms.untracked.string('genParticlePlusISRANDFSRWeights.root')
)







# Main path
process.systAna = cms.Path(
       process.printGenParticles
      *process.isrWeight
      *process.fsrWeight
      *process.isrGammaWeight
     
)

process.end = cms.EndPath(process.systematicsAnalyzer
                          * process.Output
                          )

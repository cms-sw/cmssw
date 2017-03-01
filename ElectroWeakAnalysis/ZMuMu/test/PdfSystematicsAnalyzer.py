import FWCore.ParameterSet.Config as cms

## produced the weigths according to the given pdf sets... After one needs to run the zPdfUnc.py and EvalPdfUnc.py

#### intented to run with that prescription:
### cmsrel CMSSW_3_3_X
### cd CMSSW_3_3_X/src
### addpkg ElectroWeakAnalysis/Utilities V00-01-07
###  addpkg MuonAnalysis/MomentumScaleCalibration V00-03-03
### scram setup lhapdffull
### scram b ToolUpdated
### emacs -nw ElectroWeakAnalysis/Utilities/BuildFile .... to change the build file ( Comment the <use name=lhapdf> and Uncomment the <use name=lhapdffull>)
### scram b
### cd ElectroWeakAnalysis/Utilities/test/
### cmsenv
#### cmsRun PdfSystematicsAnalyzer.py 


# Process name
process = cms.Process("PDFANA")




# Max events and printouts
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.MessageLogger.cerr.threshold = ''
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

#process.options = cms.untracked.PSet(
#  wantSummary = cms.untracked.bool(True)
#)


# Input files (on disk)
process.source = cms.Source("PoolSource",
      debugVerbosity = cms.untracked.uint32(0),
      debugFlag = cms.untracked.bool(False),
      fileNames = cms.untracked.vstring(

    #"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/Zmumu7TeV/GEN-SIM_RECO/0014/0ABB0814-C082-DE11-9AB7-003048D4767C.root",
#"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/Zmumu7TeV/GEN-SIM_RECO/0014/38980FEC-C182-DE11-A3B5-003048D4767C.root",
#"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/Zmumu7TeV/GEN-SIM_RECO/0014/3AF703B9-AE82-DE11-9656-0015172C0925.root",
#"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/Zmumu7TeV/GEN-SIM_RECO/0014/46854F8E-BC82-DE11-80AA-003048D47673.root",
#"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/Zmumu7TeV/GEN-SIM_RECO/0014/8025F9B0-AC82-DE11-8C28-0015172560C6.root",
#"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/Zmumu7TeV/GEN-SIM_RECO/0014/88DDF58E-BC82-DE11-ADD8-003048D47679.root",
#"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/Zmumu7TeV/GEN-SIM_RECO/0014/9A115324-BB82-DE11-9C66-001517252130.root",
#"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/Zmumu7TeV/GEN-SIM_RECO/0014/FC279CAC-AD82-DE11-BAAA-001517357D36.root")
)
# Produce PDF weights (maximum is 3)
process.pdfWeights = cms.EDProducer("PdfWeightProducer",
      PdfInfoTag = cms.untracked.InputTag("VtxSmeared"),
      PdfSetNames = cms.untracked.vstring(
              "cteq65.LHgrid", # 21 members
              "MRST2006nnlo.LHgrid" # 31 members
             , "MRST2007lomod.LHgrid" # 1 member
      )
)

## other three pdf sets
# Produce PDF weights (maximum is 3)
#process.pdfWeights = cms.EDProducer("PdfWeightProducer",
#      PdfInfoTag = cms.untracked.InputTag("VtxSmeared"),
#      PdfSetNames = cms.untracked.vstring(
#              "cteq61.LHgrid", # 21 members
#              "MRST2004nlo.LHgrid" # 1 members
#             , "MRST2004nnlo.LHgrid" # 1 member
#      )
#)




# Save PDF weights in the output file 
process.load("Configuration.EventContent.EventContent_cff")
process.MyEventContent = cms.PSet( 
      outputCommands = process.AODSIMEventContent.outputCommands
)
process.MyEventContent.outputCommands.extend(
      cms.untracked.vstring('drop *',
                            'keep *_genParticles_*_*',
                            'keep *_pdfWeights_*_*',
                           # 'keep *_MRST2007lomodewkPdfWeights_*_*',                                        'keep
# *_cteq6mLHewkPdfWeights_*_*',
 #                            'keep *_MRST2007lomodewkPdfWeights_*_*',                                        'kee
#p *_MRST2004nloewkPdfWeights_*_*',
 #                            'keep *_genEventWeight_*_*'
                            )
)

# Output (optionaly filtered by path)
process.pdfOutput = cms.OutputModule("PoolOutputModule",
    process.MyEventContent,
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pdfana')
    ),
    fileName = cms.untracked.string('genParticlePlusCteq65AndMRST06NNLOAndMSTW2007LOmodWeigths.root')
)







# Selector and parameters
# WMN fast selector (use W candidates in this example)
#process.load("ElectroWeakAnalysis.WMuNu.WMuNuSelection_cff")

# Collect uncertainties for rate and acceptance
process.pdfSystematics = cms.EDFilter("PdfSystematicsAnalyzer",
      SelectorPath = cms.untracked.string('pdfana'),
      PdfWeightTags = cms.untracked.VInputTag(
              "pdfWeights:cteq65"
            , "pdfWeights:MRST2006nnlo"
            , "pdfWeights:MRST2007lomod"
      )
)

# Main path
process.pdfana = cms.Path(
       process.pdfWeights
  
)

process.end = cms.EndPath(process.pdfSystematics * process.pdfOutput)

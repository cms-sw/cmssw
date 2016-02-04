import FWCore.ParameterSet.Config as cms

## Define the process
process = cms.Process("Top")

## Define the input sample
process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(
    'file:patTuple.root'
  )
)
## restrict the number of events for testing
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

from PhysicsTools.PatExamples.samplesCERN_cff import *
##process.source.fileNames = muonSKIM        ## ATTENTION these samples are NOT available on castor
##process.source.fileNames = simulationQCD
process.source.fileNames = simulationWjets
##process.source.fileNames = simulationZjets
##process.source.fileNames = simulationTtbar

## Define the TFileService
process.TFileService = cms.Service("TFileService",
##fileName = cms.string('analyzePatTopSelection.root')
##fileName = cms.string('analyzePatTopSelection_qcd.root')
##fileName = cms.string('analyzePatTopSelection_wjets.root')
##fileName = cms.string('analyzePatTopSelection_zjets.root')
fileName = cms.string('analyzePatTopSelection_ttbar.root')
)

## ----------------------------------------------------------------
## Apply object selection according to TopPAG reference selection
## for ICHEP 2010. This will result in 5 additional collections:
##
## * goodJets
## * vetoElecs
## * vetoMuons
## * looseMuons
## * tightMuons
##
## Have a look ont the cff file to learn more about the exact
## selection citeria.
## ----------------------------------------------------------------
process.load("PhysicsTools.PatExamples.topObjectSelection_cff")
process.topObjectProduction = cms.Path(
    process.topObjectSelection
)

## ----------------------------------------------------------------
## Define the steps for the TopPAG reference selection for ICHEP
## 2010. Have a look at the WorkBookPATExampleTopQuarks. These
## are event selections. They make use of the object selections
## applied in the step above.
## ----------------------------------------------------------------

## Trigger bit (HLT_mu9)
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
process.step1  = hltHighLevel.clone(TriggerResultsTag = "TriggerResults::HLT", HLTPaths = ["HLT_Mu9"])
## Vertex requirement
process.step2  = cms.EDFilter("VertexSelector", src = cms.InputTag("offlinePrimaryVertices"), cut = cms.string("!isFake && ndof > 4 && abs(z) < 15 && position.Rho < 2"), filter = cms.bool(True))
## Exact one tight muon
from PhysicsTools.PatAlgos.selectionLayer1.muonCountFilter_cfi import *
process.step3a = countPatMuons.clone(src = 'tightMuons', minNumber = 1, maxNumber = 1)
## Exact one loose muon
process.step3b = countPatMuons.clone(src = 'looseMuons', minNumber = 1, maxNumber = 1)
## Veto on additional muons
process.step4  = countPatMuons.clone(src = 'vetoMuons' , maxNumber = 1)
## Veto on additional electrons
from PhysicsTools.PatAlgos.selectionLayer1.electronCountFilter_cfi import *
process.step5  = countPatMuons.clone(src = 'vetoElecs' , maxNumber = 0)
## Different jet multiplicity selections
from PhysicsTools.PatAlgos.selectionLayer1.jetCountFilter_cfi import *
process.step6a = countPatJets.clone(src = 'goodJets'   , minNumber = 1)
process.step6b = countPatJets.clone(src = 'goodJets'   , minNumber = 2)
process.step6c = countPatJets.clone(src = 'goodJets'   , minNumber = 3)
process.step7  = countPatJets.clone(src = 'goodJets'   , minNumber = 4)


## ----------------------------------------------------------------
## Define monitoring modules for the event selection. You should
## few this only as an example for an analyses technique including
## full CMSSW features, not as a complete analysis.
## ----------------------------------------------------------------

from PhysicsTools.PatExamples.PatTopSelectionAnalyzer_cfi import *
process.monStart  = analyzePatTopSelection.clone(jets='goodJets')
process.monStep1  = analyzePatTopSelection.clone(jets='goodJets')
process.monStep2  = analyzePatTopSelection.clone(jets='goodJets')
process.monStep3a = analyzePatTopSelection.clone(muons='tightMuons', jets='goodJets')
process.monStep4  = analyzePatTopSelection.clone(muons='vetoMuons' , jets='goodJets')
process.monStep5  = analyzePatTopSelection.clone(muons='vetoMuons', elecs='vetoElecs', jets='goodJets')
process.monStep6a = analyzePatTopSelection.clone(muons='vetoMuons', elecs='vetoElecs', jets='goodJets')
process.monStep6b = analyzePatTopSelection.clone(muons='vetoMuons', elecs='vetoElecs', jets='goodJets')
process.monStep6c = analyzePatTopSelection.clone(muons='vetoMuons', elecs='vetoElecs', jets='goodJets')
process.monStep7  = analyzePatTopSelection.clone(muons='vetoMuons', elecs='vetoElecs', jets='goodJets')


## ----------------------------------------------------------------
## Define the analysis paths: we define two selection paths to
## monitor the cutflow according to the TopPAG reference selection
## for ICHEP 2010. All necessary object collections have been pro-
## duced in the cms.Path topObjectProduction before hand. The out-
## put report is switched on to get a quick overview of the number
## number of events after each selection step.
## ----------------------------------------------------------------

## Switch output report on
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

## Define loose event selection path
process.looseEventSelection = cms.Path(
    process.step1      *
    process.step2      *
    process.step3b     *
    process.step4      *
    process.step5      *
    process.step6a     *
    process.step6b     *
    process.step6c
    )

## Define tight event selection path
process.tightEventSelection = cms.Path(
    process.monStart   *
    process.step1      *
    process.monStep1   *
    process.step2      *
    process.monStep2   *
    process.step3a     *
    process.monStep3a  *
    process.step4      *
    process.monStep4   *
    process.step5      *
    process.monStep5   *
    process.step6a     *
    process.monStep6a  *
    process.step6b     *
    process.monStep6b  *
    process.step6c     *
    process.monStep6c  *
    process.step7      *
    process.monStep7
    )

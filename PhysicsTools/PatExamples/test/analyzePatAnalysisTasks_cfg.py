import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(
    "file:patTuple_PATandPF2PAT.root"
  )
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

#from FWCore.MessageLogger.MessageLogger_cfi import MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
#process.MessageLogger.categories.append('hint1')
#process.MessageLogger.categories.append('hint2')
#process.MessageLogger.categories.append('hint3')

#################
#               #
# EXERCISE 1    #
#               #
#################

process.jecAnalyzer = cms.EDAnalyzer("WrappedEDAnalysisTasksAnalyzerJEC",
  Jets = cms.InputTag("selectedPatJetsPFlow"), 
  jecLevel=cms.string("L3Abs"),
  jecSetLabel= cms.string('patJetCorrFactors'),
  outputFileName=cms.string("jecAnalyzerOutput")
)

process.p_jec = cms.Path(process.jecAnalyzer)

#process.jecAnalyzerRel=process.jecAnalyzer.clone(jecLevel="L2Relative")
#process.p_jec.__iadd__(process.jecAnalyzerRel)
#process.jecAnalyzerNon=process.jecAnalyzer.clone(jecLevel="Uncorrected")
#process.p_jec.__iadd__(process.jecAnalyzerNon)

#################
#               #
# EXERCISE 2    #
#               #
#################

process.TFileService = cms.Service("TFileService",
  fileName = cms.string('TFileServiceOutput.root')
)

process.btagAnalyzer = cms.EDAnalyzer("WrappedEDAnalysisTasksAnalyzerBTag",
                                      Jets = cms.InputTag("selectedPatJetsPFlow"),    
                                      bTagAlgo=cms.string('trackCounting'),
                                      bins=cms.uint32(100),
                                      lowerbin=cms.double(0.),
                                      upperbin=cms.double(10.)
)

process.p_btag = cms.Path(process.btagAnalyzer)
#process.btagAnalyzerTCHP=process.btagAnalyzer.clone(bTagAlgo="trackCountingHighPurBJetTags")
#process.p_btag.__iadd__(process.btagAnalyzerTCHP)


#################
#               #
# EXERCISE 3    #
#               #
#################

## The MET Uncertainty tool needs some more things to be there:
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = cms.string( autoCond[ 'startup' ] )
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("PhysicsTools.PatAlgos.patSequences_cff")

#Applying the MET Uncertainty tools
from PhysicsTools.PatUtils.tools.metUncertaintyTools import runMEtUncertainties
runMEtUncertainties(process, electronCollection = cms.InputTag("selectedPatElectronsPFlow"), jetCollection="selectedPatJetsPFlow", muonCollection = cms.InputTag("selectedPatMuonsPFlow"), tauCollection = cms.InputTag("selectedPatTausPFlow") )


#process.shiftedPatJetsEnUp=process.shiftedPatJetsPFlowEnUpForCorrMEt.clone(shiftBy=cms.double(2), src="selectedPatJetsPFlow")
#process.jecAnalyzerEnUp=process.jecAnalyzer.clone(Jets = cms.InputTag("shiftedPatJetsEnUp"))
#process.p_jec.__iadd__( process.shiftedPatJetsEnUp *  process.jecAnalyzerEnUp)

#################
#               #
# EXERCISE 4    #
#               #
#################

process.patJPsiCandidates = cms.EDProducer("PatJPsiProducer",
                                           muonSrc = cms.InputTag("selectedPatMuonsPFlow")
                                           )

process.myGoodJPsiCandidates = cms.EDFilter("PATCompositeCandidateSelector",
                                       src = cms.InputTag("patJPsiCandidates"),
                                       cut = cms.string(" abs( mass() -  3.097) < 100 ") # "userFloat('dR') < 100"
                                       )

process.selectEventsWithGoodJspiCand = cms.EDFilter("PATCandViewCountFilter",
                                                minNumber = cms.uint32(1),
                                                maxNumber = cms.uint32(100000),
                                                src = cms.InputTag("myGoodJPsiCandidates")
                                                )

#process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.DRHisto_JPsiCands= cms.EDAnalyzer( "CandViewHistoAnalyzer",
                                           src = cms.InputTag("patJPsiCandidates"),
                                           #weights=cms.untracked.InputTag(pileupweight),
                                           histograms = cms.VPSet(
                                                    cms.PSet(
                                                        min          = cms.untracked.double(       0.),
                                                        max          = cms.untracked.double(      10.),
                                                        nbins        = cms.untracked.int32 (       60),
                                                        name         = cms.untracked.string('DR_JPsi'),
                                                        description  = cms.untracked.string('DR of JPsi Candidate; DR(#mu1,#mu2); number of JPsi Cands'),
                                                        lazyParsing  = cms.untracked.bool(True),
                                                        plotquantity = cms.untracked.string("userFloat('dR')")
                                                        )   
                                                    )
                                           )

#process.p_jpsi= cms.Path(process.patJPsiCandidates  * process.DRHisto_JPsiCands * process.myGoodJPsiCandidates * process.selectEventsWithGoodJspiCand)

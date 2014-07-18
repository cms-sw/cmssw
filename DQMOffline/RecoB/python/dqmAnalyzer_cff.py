import FWCore.ParameterSet.Config as cms

#not useful anymore for b-tagging but used in some other sequences
from JetMETCorrections.Configuration.JetCorrectionServices_cff import ak5PFL2L3,ak5PFL2Relative,ak5PFL3Absolute

#JEC for CHS
from JetMETCorrections.Configuration.JetCorrectionServices_cff import ak5PFCHSL1Fastjet, ak5PFCHSL2Relative, ak5PFCHSL3Absolute, ak5PFCHSResidual, ak5PFCHSL1FastL2L3, ak5PFCHSL1FastL2L3Residual
newak5PFCHSL1Fastjet = ak5PFCHSL1Fastjet.clone(algorithm = 'AK5PFchs')
newak5PFCHSL2Relative = ak5PFCHSL2Relative.clone(algorithm = 'AK5PFchs')
newak5PFCHSL3Absolute = ak5PFCHSL3Absolute.clone(algorithm = 'AK5PFchs')
newak5PFCHSResidual = ak5PFCHSResidual.clone(algorithm = 'AK5PFchs')

newak5PFCHSL1FastL2L3 = ak5PFCHSL1FastL2L3.clone(correctors = cms.vstring('newak5PFCHSL1Fastjet','newak5PFCHSL2Relative','newak5PFCHSL3Absolute'))
newak5PFCHSL1FastL2L3Residual = ak5PFCHSL1FastL2L3Residual.clone(correctors = cms.vstring('newak5PFCHSL1Fastjet','newak5PFCHSL2Relative','newak5PFCHSL3Absolute','newak5PFCHSResidual'))

#Needed only for fastsim, why?
ak5PFCHSL1Fastjet.algorithm = 'AK5PFchs'
ak5PFCHSL2Relative.algorithm = 'AK5PFchs'
ak5PFCHSL3Absolute.algorithm = 'AK5PFchs'
ak5PFCHSResidual.algorithm = 'AK5PFchs'

######### DATA ############
from DQMOffline.RecoB.bTagAnalysisData_cfi import *
bTagAnalysis.finalizePlots = False
bTagAnalysis.finalizeOnly = False
bTagAnalysis.ptRanges = cms.vdouble(0.0)
#Residual correction will be added inside the c++ code only for data (checking the presence of genParticles collection), not explicit here as this sequence also ran on MC FullSim
bTagAnalysis.doJetID = True
bTagAnalysis.doJEC = True
bTagAnalysis.JECsource = cms.string("newak5PFCHSL1FastL2L3") 
bTagPlotsDATA = cms.Sequence(bTagAnalysis)

########## MC ############
#Matching
from PhysicsTools.JetMCAlgos.CaloJetsMCFlavour_cfi import *
AK5byRef.jets = cms.InputTag("ak5PFJetsCHS")
#Get gen jet collection for real jets
ak5GenJetsForPUid = cms.EDFilter("GenJetSelector",
                                 src = cms.InputTag("ak5GenJets"),
                                 cut = cms.string('pt > 8.'),
                                 filter = cms.bool(False)
                                 )
#do reco gen - reco matching
from PhysicsTools.PatAlgos.mcMatchLayer0.jetMatch_cfi import patJetGenJetMatch
newpatJetGenJetMatch = patJetGenJetMatch.clone(
    src = cms.InputTag("ak5PFJetsCHS"),
    matched = cms.InputTag("ak5GenJetsForPUid"),
    maxDeltaR = cms.double(0.25),
    resolveAmbiguities = cms.bool(True)
)

# Module execution for MC
from Validation.RecoB.bTagAnalysis_cfi import *
bTagValidation.finalizePlots = False
bTagValidation.finalizeOnly = False
bTagValidation.jetMCSrc = 'AK5byValAlgo'
bTagValidation.ptRanges = cms.vdouble(0.0)
bTagValidation.etaRanges = cms.vdouble(0.0)
bTagValidation.doJetID = True
bTagValidation.doJEC = True
bTagValidation.JECsource = cms.string("newak5PFCHSL1FastL2L3")
bTagValidation.doPUid = cms.bool(True)
bTagValidation.genJetsMatched = cms.InputTag("newpatJetGenJetMatch")
#to run on fastsim
prebTagSequenceMC = cms.Sequence(ak5GenJetsForPUid*newpatJetGenJetMatch*myPartons*AK5Flavour)
bTagPlotsMC = cms.Sequence(bTagValidation)

#to run on fullsim in the validation sequence, all histograms produced in the dqmoffline sequence
bTagValidationNoall = bTagValidation.clone(flavPlots="noall")
bTagPlotsMCbcl = cms.Sequence(myPartons*AK5Flavour*bTagValidationNoall)

import FWCore.ParameterSet.Config as cms

#not useful anymore for b-tagging but used in some other sequences
from JetMETCorrections.Configuration.JetCorrectionServices_cff import ak4PFL2L3,ak4PFL2Relative,ak4PFL3Absolute

#JEC for CHS
from JetMETCorrections.Configuration.JetCorrectionServices_cff import ak4PFCHSL1Fastjet, ak4PFCHSL2Relative, ak4PFCHSL3Absolute, ak4PFCHSResidual, ak4PFCHSL1FastL2L3, ak4PFCHSL1FastL2L3Residual
newak4PFCHSL1Fastjet = ak4PFCHSL1Fastjet.clone(algorithm = 'AK5PFchs')
newak4PFCHSL2Relative = ak4PFCHSL2Relative.clone(algorithm = 'AK5PFchs')
newak4PFCHSL3Absolute = ak4PFCHSL3Absolute.clone(algorithm = 'AK5PFchs')
newak4PFCHSResidual = ak4PFCHSResidual.clone(algorithm = 'AK5PFchs')

newak4PFCHSL1FastL2L3 = ak4PFCHSL1FastL2L3.clone(correctors = cms.vstring('newak4PFCHSL1Fastjet','newak4PFCHSL2Relative','newak4PFCHSL3Absolute'))
newak4PFCHSL1FastL2L3Residual = ak4PFCHSL1FastL2L3Residual.clone(correctors = cms.vstring('newak4PFCHSL1Fastjet','newak4PFCHSL2Relative','newak4PFCHSL3Absolute','newak4PFCHSResidual'))

#Needed only for fastsim, why?
ak4PFCHSL1Fastjet.algorithm = 'AK5PFchs'
ak4PFCHSL2Relative.algorithm = 'AK5PFchs'
ak4PFCHSL3Absolute.algorithm = 'AK5PFchs'
ak4PFCHSResidual.algorithm = 'AK5PFchs'

######### DATA ############
from DQMOffline.RecoB.bTagAnalysisData_cfi import *
bTagAnalysis.finalizePlots = False
bTagAnalysis.finalizeOnly = False
bTagAnalysis.ptRanges = cms.vdouble(0.0)
#Residual correction will be added inside the c++ code only for data (checking the presence of genParticles collection), not explicit here as this sequence also ran on MC FullSim
bTagAnalysis.doJetID = True
bTagAnalysis.doJEC = True
bTagAnalysis.JECsource = cms.string("newak4PFCHSL1FastL2L3") 
bTagPlotsDATA = cms.Sequence(bTagAnalysis)

########## MC ############
#Matching
from PhysicsTools.JetMCAlgos.CaloJetsMCFlavour_cfi import *
AK4byRef.jets = cms.InputTag("ak4PFJetsCHS")
#Get gen jet collection for real jets
ak4GenJetsForPUid = cms.EDFilter("GenJetSelector",
                                 src = cms.InputTag("ak4GenJets"),
                                 cut = cms.string('pt > 8.'),
                                 filter = cms.bool(False)
                                 )
#do reco gen - reco matching
from PhysicsTools.PatAlgos.mcMatchLayer0.jetMatch_cfi import patJetGenJetMatch
newpatJetGenJetMatch = patJetGenJetMatch.clone(
    src = cms.InputTag("ak4PFJetsCHS"),
    matched = cms.InputTag("ak4GenJetsForPUid"),
    maxDeltaR = cms.double(0.25),
    resolveAmbiguities = cms.bool(True)
)

# Module execution for MC
from Validation.RecoB.bTagAnalysis_cfi import *
bTagValidation.finalizePlots = False
bTagValidation.finalizeOnly = False
bTagValidation.jetMCSrc = 'AK4byValAlgo'
bTagValidation.ptRanges = cms.vdouble(0.0)
bTagValidation.etaRanges = cms.vdouble(0.0)
bTagValidation.doJetID = True
bTagValidation.doJEC = True
bTagValidation.JECsource = cms.string("newak4PFCHSL1FastL2L3")
bTagValidation.doPUid = cms.bool(True)
bTagValidation.genJetsMatched = cms.InputTag("newpatJetGenJetMatch")
#to run on fastsim
prebTagSequenceMC = cms.Sequence(ak4GenJetsForPUid*newpatJetGenJetMatch*myPartons*AK4Flavour)
bTagPlotsMC = cms.Sequence(bTagValidation)

#to run on fullsim in the validation sequence, all histograms produced in the dqmoffline sequence
bTagValidationNoall = bTagValidation.clone(flavPlots="noall")
bTagPlotsMCbcl = cms.Sequence(myPartons*AK4Flavour*bTagValidationNoall)

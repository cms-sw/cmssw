import FWCore.ParameterSet.Config as cms

from DQMOffline.RecoB.bTagAnalysisData_cfi import *
bTagAnalysis.finalizePlots = False
bTagAnalysis.finalizeOnly = False
bTagAnalysis.ptRanges = cms.vdouble(0.0)
bTagPlotsDATA = cms.Sequence(bTagAnalysis)

########## MC ############
#Matching
from PhysicsTools.JetMCAlgos.CaloJetsMCFlavour_cfi import *
AK5byRef.jets = cms.InputTag("ak5PFJetsCHS")

# Module execution for MC
from Validation.RecoB.bTagAnalysis_cfi import *
bTagValidation.finalizePlots = False
bTagValidation.finalizeOnly = False
bTagValidation.jetMCSrc = 'AK5byValAlgo'
bTagValidation.ptRanges = cms.vdouble(0.0)
bTagValidation.etaRanges = cms.vdouble(0.0)
#to run on fastsim
bTagPlotsMC = cms.Sequence(myPartons*AK5Flavour*bTagValidation)

#to run on fullsim in the validation sequence, all histograms produced in the dqmoffline sequence
bTagValidationNoall = bTagValidation.clone(flavPlots="noall")
bTagPlotsMCbcl = cms.Sequence(myPartons*AK5Flavour*bTagValidationNoall)
